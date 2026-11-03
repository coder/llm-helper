// Package-local activation: the panel's wrapper around the shared control
// layer. What changes here is only how the result is reported -- warnings come
// back as data instead of going to a terminal.
package web

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/coder/vllm-helper/internal/control"
	"github.com/coder/vllm-helper/internal/profile"
)

// activate switches this machine to a saved profile.
//
// The work itself is control's: the panel must not have its own idea of what
// switching means, or the same profile would serve differently depending on
// whether a browser or a terminal asked for it.
func (s *Server) activate(ctx context.Context, name string) ([]string, error) {
	p, err := profile.Load(s.cfg.Dirs, name)
	if err != nil {
		return nil, err
	}
	if err := control.WriteActiveEnv(s.cfg, p); err != nil {
		return nil, err
	}

	// Plugins write to an io.Writer rather than assuming a terminal, so
	// their warnings can be captured and shown in the panel.
	var pluginOut bytes.Buffer
	restart := func() error { return s.privilegedCall(ctx, s.svc.Restart) }
	err = control.NotifyAndRestart(ctx, s.cfg, p, &pluginOut, restart)

	return warningsFrom(&pluginOut), err
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
