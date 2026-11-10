package web

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"os/user"
	"strconv"
	"time"

	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/detect"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/service"
)

// run executes a command and reports only whether it succeeded. Used for
// capability probes, where the output doesn't matter.
func run(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	return cmd.Run()
}

func currentUser() string {
	if u, err := user.Current(); err == nil {
		return u.Username
	}
	return "youruser"
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// statusResponse is what the panel polls. Deliberately assembled field by
// field rather than marshalling internal structs: config and profiles both
// carry credentials, and an explicit DTO is the only shape where adding a
// secret has to be a conscious act.
type statusResponse struct {
	Instances []instanceStatus `json:"instances"`
	// Default is the instance Coder offers first.
	Default string `json:"default,omitempty"`

	Privileged bool   `json:"privileged"`
	PrivReason string `json:"priv_reason,omitempty"`

	Build string `json:"build"`
	Host  string `json:"host"`
}

// instanceStatus is one running model.
type instanceStatus struct {
	Profile    string `json:"profile"`
	Model      string `json:"model"`
	Port       string `json:"port"`
	State      string `json:"state"`
	Healthy    bool   `json:"healthy"`
	HTTPStatus int    `json:"http_status"`
	Crashed    bool   `json:"crashed"`
	Restarts   int    `json:"restarts"`
	FailReason string `json:"fail_reason,omitempty"`
	LastLog    string `json:"last_log,omitempty"`

	// Phase collapses the above into the one thing a person actually wants
	// to know. The distinction that matters is loading vs failing: both
	// leave the endpoint dead, and telling someone to wait for a server
	// that is crash-looping wastes their afternoon.
	Phase string `json:"phase"`

	// Cluster is set while a recipe cluster serves the instance. The panel
	// reports it but cannot drive it, so its actions are refused and
	// ControlReason says what to use instead.
	Cluster       bool   `json:"cluster"`
	ControlReason string `json:"control_reason,omitempty"`

	Default bool `json:"default"`
}

func (s *Server) handleStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.status())
}

func (s *Server) handleInstances(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.instances())
}

func (s *Server) status() statusResponse {
	// Re-probe only while we believe privilege is missing. Probing when we
	// have it would refresh the sudo timestamp on every poll and keep it
	// alive indefinitely, which is not the panel's decision to make.
	privileged, privReason := s.privilege()
	if !privileged {
		privileged, privReason = checkPrivilege(s.cfg)
		s.setPrivilege(privileged, privReason)
	}
	r := statusResponse{
		Instances:  s.instances(),
		Privileged: privileged,
		PrivReason: privReason,
		Build:      s.build,
		Host:       hostname(),
	}
	for _, i := range r.Instances {
		if i.Default {
			r.Default = i.Profile
		}
	}
	return r
}

func (s *Server) instances() []instanceStatus {
	list, _ := control.ListInstances(s.cfg)
	names := make([]string, 0, len(list))
	for _, inst := range list {
		names = append(names, inst.Name)
	}
	def := control.DefaultInstance(s.cfg, names)
	out := make([]instanceStatus, 0, len(list))
	for _, inst := range list {
		out = append(out, s.instanceStatus(inst, inst.Name == def))
	}
	return out
}

func (s *Server) instanceStatus(inst *control.Instance, isDefault bool) instanceStatus {
	cluster := s.cluster(inst.Name)
	var health func() service.Health
	var lastLogs func(int) string
	if cluster != nil {
		health, lastLogs = cluster.Health, cluster.LastLogs
	} else {
		svc := s.service(inst.Name)
		health, lastLogs = svc.Health, svc.LastLogs
	}
	h := health()
	r := instanceStatus{
		Profile:    inst.Name,
		Model:      "unknown",
		Port:       inst.Port,
		State:      h.State,
		Crashed:    h.Crashed,
		Restarts:   h.NRestarts,
		FailReason: h.Result,
		Default:    isDefault,
	}
	if m := inst.Env.Get("MODEL"); m != "" {
		r.Model = m
	}
	if cluster != nil {
		r.Cluster, r.ControlReason = true, clusterControlReason
	}
	// Probing a unit that isn't running can only fail: engines bind the
	// port after the weights are in memory.
	running := h.State == "active" || h.State == "activating"
	if running {
		r.HTTPStatus, r.Healthy = s.probe(inst)
	}
	switch {
	case r.Healthy:
		r.Phase = "serving"
	case h.Crashed:
		r.Phase = "failing"
		r.LastLog = control.LastMeaningfulLine(lastLogs(40))
	case running:
		r.Phase = "loading"
	default:
		r.Phase = "stopped"
	}
	return r
}

// profileResponse describes one saved model. Note the absence of the raw
// profile file: it is an env file a user may have hand-edited, and anything
// they put in it would end up in a browser.
type profileResponse struct {
	Name        string `json:"name"`
	Model       string `json:"model"`
	DisplayName string `json:"display_name"`
	MaxModelLen string `json:"max_model_len"`
	MemUtil     string `json:"gpu_memory_utilization"`
	ExtraArgs   string `json:"extra_args"`
	Status      string `json:"status"`
	Running     bool   `json:"running"`
}

func (s *Server) handleProfiles(w http.ResponseWriter, _ *http.Request) {
	profiles, err := profile.List(s.cfg.Dirs)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	running := map[string]bool{}
	for _, n := range control.LiveProfiles(s.cfg) {
		running[n] = true
	}
	out := make([]profileResponse, 0, len(profiles))
	for _, p := range profiles {
		out = append(out, profileResponse{
			Name:        p.Name,
			Model:       p.Model(),
			DisplayName: p.DisplayName(),
			MaxModelLen: p.MaxModelLen(),
			MemUtil:     p.File.Get("GPU_MEMORY_UTILIZATION"),
			ExtraArgs:   p.ExtraArgs(),
			Status:      p.Status(),
			Running:     running[p.Name],
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// machineResponse is the hardware the panel draws its memory rail from.
//
// BudgetBytes and BudgetGB shadow the embedded pair deliberately: the probe
// reports what this machine could hold, while every verdict on the models page
// is measured against the configured budget. Two numbers for one limit, sitting
// beside each other on the same page, is worse than either.
//
// PoolBytes/Pool say which memory the budget was measured against, because on
// a discrete GPU that is VRAM and drawing it over installed RAM would put the
// marker in a meaningless place.
type machineResponse struct {
	detect.Hardware
	BudgetBytes int64 `json:"budget_bytes"`
	BudgetGB    int   `json:"budget_gb"`
	// BudgetKnown separates a measured budget of 0, where nothing fits,
	// from one that could not be worked out.
	BudgetKnown    bool   `json:"budget_known"`
	PoolBytes      int64  `json:"pool_bytes"`
	Pool           string `json:"pool"`
	AvailableMemMB int    `json:"available_mem_mb"`
}

func (s *Server) handleMachine(w http.ResponseWriter, _ *http.Request) {
	hw := detect.Probe()
	poolBytes, pool := hw.Pool()
	budget := s.budget()
	writeJSON(w, http.StatusOK, machineResponse{
		Hardware:       hw,
		BudgetBytes:    int64(budget.GB * 1e9),
		BudgetGB:       int(budget.GB),
		BudgetKnown:    budget.Known,
		PoolBytes:      poolBytes,
		Pool:           pool,
		AvailableMemMB: detect.AvailableMemMB(),
	})
}

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	n := 200
	if v := r.URL.Query().Get("n"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 && parsed <= 2000 {
			n = parsed
		}
	}
	name := r.URL.Query().Get("instance")
	if name == "" {
		list, _ := control.ListInstances(s.cfg)
		if len(list) != 1 {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "name an instance with ?instance="})
			return
		}
		name = list[0].Name
	}
	if profile.ValidInstanceName(name) != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid instance name"})
		return
	}
	if c := s.cluster(name); c != nil {
		writeJSON(w, http.StatusOK, map[string]string{"logs": c.LastLogs(n), "hint": c.LogHint()})
		return
	}
	svc := s.service(name)
	writeJSON(w, http.StatusOK, map[string]string{"logs": svc.LastLogs(n), "hint": svc.LogHint()})
}

// actionResponse is the result of something that changes state.
type actionResponse struct {
	OK       bool     `json:"ok"`
	Error    string   `json:"error,omitempty"`
	Warnings []string `json:"warnings,omitempty"`
}

// action runs fn for one instance with the checks every action shares.
func (s *Server) action(w http.ResponseWriter, r *http.Request, fn func(ctx context.Context, name string) ([]string, int, error)) {
	name := r.PathValue("name")
	if err := profile.ValidInstanceName(name); err != nil {
		writeJSON(w, http.StatusBadRequest, actionResponse{Error: err.Error()})
		return
	}
	if !s.requirePrivilege(w, name) {
		return
	}
	defer s.lock(name)()
	defer s.forgetHealth(name)
	warnings, code, err := fn(r.Context(), name)
	if err != nil {
		if code == 0 {
			code = http.StatusInternalServerError
		}
		writeJSON(w, code, actionResponse{Error: err.Error(), Warnings: warnings})
		return
	}
	writeJSON(w, http.StatusOK, actionResponse{OK: true, Warnings: warnings})
}

func (s *Server) handleUp(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	opts := control.InstanceOpts{Port: q.Get("port"), MemUtil: q.Get("mem_util")}
	s.action(w, r, func(ctx context.Context, name string) ([]string, int, error) {
		return s.up(ctx, name, opts)
	})
}

func (s *Server) handleDown(w http.ResponseWriter, r *http.Request) {
	s.action(w, r, func(ctx context.Context, name string) ([]string, int, error) {
		if _, err := control.LoadInstance(s.cfg, name); err != nil {
			return nil, http.StatusNotFound, fmt.Errorf("'%s' is not running", name)
		}
		if err := s.privilegedCall(ctx, name, s.service(name).Uninstall); err != nil {
			return nil, 0, err
		}
		if err := control.RemoveInstance(s.cfg, name); err != nil {
			return nil, 0, err
		}
		var out bytes.Buffer
		control.NotifyStopped(ctx, s.cfg, name, &out)
		return warningsFrom(&out), 0, nil
	})
}

func (s *Server) handleRestart(w http.ResponseWriter, r *http.Request) {
	s.action(w, r, func(ctx context.Context, name string) ([]string, int, error) {
		inst, err := control.LoadInstance(s.cfg, name)
		if err != nil {
			return nil, http.StatusNotFound, fmt.Errorf("'%s' is not running", name)
		}
		p, err := profile.Load(s.cfg.Dirs, name)
		if err != nil {
			return nil, 0, err
		}
		if err := control.WriteInstanceEnv(s.cfg, p, inst.Opts()); err != nil {
			return nil, 0, err
		}
		return nil, 0, s.privilegedCall(ctx, name, s.service(name).Restart)
	})
}

func (s *Server) handleDefault(w http.ResponseWriter, r *http.Request) {
	s.action(w, r, func(ctx context.Context, name string) ([]string, int, error) {
		if _, err := control.LoadInstance(s.cfg, name); err != nil {
			return nil, http.StatusNotFound, fmt.Errorf("'%s' is not running", name)
		}
		s.cfg.File.Set("DEFAULT_PROFILE", name)
		if err := s.cfg.File.Write(s.cfg.Dirs.ConfigFile(), 0o644); err != nil {
			return nil, 0, err
		}
		var out bytes.Buffer
		control.NotifyDefault(ctx, s.cfg, &out)
		return warningsFrom(&out), 0, nil
	})
}

// explain turns a failed systemctl call into something actionable.
//
// The service package sends the real reason to the daemon's stderr, so what
// reaches here is a bare "exit status 1". The common cause is a sudo
// timestamp that has since expired -- the startup check passed, so the
// buttons were enabled, and by the time one was pressed the credential was
// gone. Re-checking on failure costs one fork and turns a meaningless error
// into the sudoers line that fixes it.
func (s *Server) explain(name string, err error) error {
	if ok, reason := checkPrivilege(s.cfg); !ok {
		s.setPrivilege(false, reason)
		return errors.New(reason)
	}
	if hint := s.service(name).LogHint(); hint != "" {
		return fmt.Errorf("%w -- see '%s'", err, hint)
	}
	return err
}

func (s *Server) requirePrivilege(w http.ResponseWriter, name string) bool {
	return s.requireNative(w, name) && s.requireSudo(w)
}

// requireNative refuses an action on an instance a recipe cluster serves: the
// panel's verbs drive the instance's own unit, so a restart would start a
// second server beside the cluster.
func (s *Server) requireNative(w http.ResponseWriter, name string) bool {
	if s.cluster(name) == nil {
		return true
	}
	writeJSON(w, http.StatusConflict, actionResponse{Error: clusterControlReason})
	return false
}

func (s *Server) requireSudo(w http.ResponseWriter) bool {
	if ok, _ := s.privilege(); ok {
		return true
	}
	_, reason := s.privilege()
	writeJSON(w, http.StatusForbidden, actionResponse{Error: reason})
	return false
}

// privilegedCall bounds a systemctl call in time. Nothing in the service
// package takes a context, and a systemctl that blocks would otherwise hold
// an HTTP request open indefinitely.
func (s *Server) privilegedCall(ctx context.Context, name string, fn func() error) error {
	done := make(chan error, 1)
	go func() { done <- fn() }()
	select {
	case err := <-done:
		if err != nil {
			return s.explain(name, err)
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(2 * time.Minute):
		return fmt.Errorf("systemctl did not return within two minutes")
	}
}
