// Package-local activation: the panel's wrapper around the shared control
// layer. What changes here is only how the result is reported -- warnings come
// back as data instead of going to a terminal.
package web

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
)

// up starts a saved profile as an instance beside whatever already runs.
// Starting one that already runs restarts it with the new settings.
//
// The work itself is control's: the panel must not have its own idea of what
// running a profile means, or the same profile would serve differently
// depending on whether a browser or a terminal asked for it.
func (s *Server) up(ctx context.Context, name string, opts control.InstanceOpts) ([]string, int, error) {
	if opts.MemUtil != "" {
		if f, err := strconv.ParseFloat(opts.MemUtil, 64); err != nil || f <= 0 || f > 1 {
			return nil, http.StatusBadRequest, fmt.Errorf("mem_util %q: want a fraction between 0 and 1", opts.MemUtil)
		}
	}
	if _, err := os.Stat(s.cfg.Dirs.LegacyActiveEnvFile()); err == nil {
		return nil, http.StatusConflict, errors.New("this machine still runs a model the old single-model way; run 'llm-helper migrate' first")
	}
	p, err := profile.Load(s.cfg.Dirs, name)
	if err != nil {
		return nil, http.StatusNotFound, err
	}
	if p.Generated() {
		return nil, http.StatusConflict, errors.New(clusterControlReason)
	}
	svc := s.service(name)
	if !svc.Installed() && s.cfg.ServiceScope() == "system" && os.Geteuid() != 0 {
		return nil, http.StatusConflict, fmt.Errorf("the model service isn't installed yet; run 'llm-helper up %s' once from a terminal", name)
	}
	if err := control.CheckFitFor(s.cfg, name, opts.MemUtil, p.File, s.cfg.File); err != nil {
		return nil, http.StatusConflict, err
	}
	port, err := control.AllocatePort(s.cfg, name, opts.Port)
	if err != nil {
		return nil, http.StatusConflict, err
	}
	opts.Port = port
	if err := control.WriteInstanceEnv(s.cfg, p, opts); err != nil {
		return nil, 0, err
	}
	inst, err := control.LoadInstance(s.cfg, name)
	if err != nil {
		return nil, 0, err
	}
	self, err := os.Executable()
	if err != nil {
		return nil, 0, err
	}
	install := func() error { _, err := svc.Install(self); return err }
	if err := s.privilegedCall(ctx, name, install); err != nil {
		_ = control.RemoveInstance(s.cfg, name)
		return nil, 0, err
	}

	// Plugins write to an io.Writer rather than assuming a terminal, so
	// their warnings can be captured and shown in the panel.
	var pluginOut bytes.Buffer
	restart := func() error { return s.privilegedCall(ctx, name, svc.Restart) }
	err = control.NotifyAndRestart(ctx, s.cfg, p, inst, &pluginOut, restart)
	return warningsFrom(&pluginOut), 0, err
}

// warningsFrom turns captured plugin output into lines a page can show.
func warningsFrom(buf *bytes.Buffer) []string {
	var out []string
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	return out
}

func hostname() string {
	h, err := os.Hostname()
	if err != nil {
		return ""
	}
	return filepath.Base(h)
}
