// Package web serves a small control panel for the machine this binary runs
// on: what model is loaded, whether it is healthy, and what else could be
// loaded instead.
//
// It is a control plane, not a proxy. Inference traffic goes straight to
// vLLM on its own port; nothing here sits in the path of a token. That keeps
// the server irrelevant to whether the model works, which is the property
// worth protecting -- a bug in this package must never be able to break
// inference.
package web

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/service"
	"github.com/coder/llm-helper/internal/vllm"
)

// Server holds everything the handlers need.
type Server struct {
	cfg   *config.Config
	token string

	// clusterFor reports the recipe cluster serving an instance, or nil when
	// its own unit does. The CLI owns that lookup, so it is passed in.
	clusterFor func(name string) Cluster

	// svcFor builds the unit for an instance; tests replace it.
	svcFor func(name string) service.Service

	// build identifies this binary in the UI, passed in because the CLI
	// package owns the ldflag-stamped values.
	build string

	// privileged reports whether service control can actually work. On a
	// system-scope unit the daemon needs passwordless sudo; without it the
	// UI shows the controls as unavailable instead of offering buttons
	// that fail with "exit status 1". Re-evaluated while it is false.
	mu         sync.RWMutex
	privileged bool
	privReason string

	// actionMu holds one mutex per profile name, so two requests never
	// change the same instance at once while different instances proceed.
	actionMu sync.Map

	// Health probes, reused for healthProbeTTL per instance: every probe
	// lands in the engine's access log.
	healthMu sync.Mutex
	health   map[string]healthEntry

	// install tracks a GGUF plugin install, which builds from source and runs
	// for minutes. It is deliberately not a blocking request: a POST held open
	// that long is dropped by browsers and proxies, and the page it belongs to
	// has no way to find out whether the work continued.
	installMu sync.Mutex
	install   installState
}

// installState is what a caller can learn about a background install.
type installState struct {
	Running bool
	Done    bool
	Err     string
	Log     string
	Started time.Time
}

// installStatus reports a copy of the current state.
func (s *Server) installStatus() installState {
	s.installMu.Lock()
	defer s.installMu.Unlock()
	return s.install
}

// beginInstall marks an install as started, refusing a second one. Returning
// false means one is already running, which the caller reports rather than
// queueing -- two pips writing to one site-packages is not something to do.
func (s *Server) beginInstall() bool {
	s.installMu.Lock()
	defer s.installMu.Unlock()
	if s.install.Running {
		return false
	}
	s.install = installState{Running: true, Started: time.Now()}
	return true
}

// finishInstall records the outcome for whoever polls next.
func (s *Server) finishInstall(err error, log string) {
	s.installMu.Lock()
	defer s.installMu.Unlock()
	s.install.Running = false
	s.install.Done = true
	s.install.Log = log
	if err != nil {
		s.install.Err = err.Error()
	}
}

type healthEntry struct {
	at   time.Time
	code int
	ok   bool
}

// healthProbeTTL bounds how often an endpoint is asked how it is. The panel
// polls in seconds and a model loads in minutes.
const healthProbeTTL = 5 * time.Second

// probe asks an instance's endpoint, or returns a recent answer instead. The
// lock is held across the request so concurrent callers share one probe.
func (s *Server) probe(inst *control.Instance) (int, bool) {
	s.healthMu.Lock()
	defer s.healthMu.Unlock()
	if e, ok := s.health[inst.Name]; ok && time.Since(e.at) < healthProbeTTL {
		return e.code, e.ok
	}
	code, ok := inst.Probe()
	if s.health == nil {
		s.health = map[string]healthEntry{}
	}
	s.health[inst.Name] = healthEntry{at: time.Now(), code: code, ok: ok}
	return code, ok
}

// forgetHealth drops an instance's cached probe, so the next status request
// asks the process that is actually running now.
func (s *Server) forgetHealth(name string) {
	s.healthMu.Lock()
	defer s.healthMu.Unlock()
	delete(s.health, name)
}

// lock serializes changes to one instance.
func (s *Server) lock(name string) func() {
	m, _ := s.actionMu.LoadOrStore(name, &sync.Mutex{})
	mu := m.(*sync.Mutex)
	mu.Lock()
	return mu.Unlock
}

// service is the unit that runs an instance.
func (s *Server) service(name string) service.Service {
	if s.svcFor != nil {
		return s.svcFor(name)
	}
	// NonInteractive always: a server never has anyone to answer a sudo
	// prompt, and a bare sudo would block with a request still open.
	return service.Service{
		Name:           service.InstanceName(name),
		Scope:          s.cfg.ServiceScope(),
		NonInteractive: true,
	}
}

// caps describes what the installed vLLM will accept, so a model card's flags
// can be checked before they reach a profile.
//
// Probe caches its answer on disk keyed by vLLM version, so this is cheap
// after the first call and correct across an upgrade.
func (s *Server) caps() vllm.Caps {
	return vllm.Probe(s.cfg.VLLMBin(), s.cfg.Dirs.State)
}

// privilege reports the cached answer.
func (s *Server) privilege() (bool, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.privileged, s.privReason
}

func (s *Server) setPrivilege(ok bool, reason string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.privileged, s.privReason = ok, reason
}

// New builds a server for this machine.
func New(cfg *config.Config, build string) (*Server, error) {
	token, err := loadOrCreateToken(cfg.Dirs.State)
	if err != nil {
		return nil, err
	}
	s := &Server{
		cfg:   cfg,
		build: build,
		token: token,
	}
	ok, reason := checkPrivilege(cfg)
	s.setPrivilege(ok, reason)
	return s, nil
}

// Cluster is what the panel can learn about a recipe cluster: enough to report
// it truthfully, nothing that drives it. Without this the panel read the idle
// native unit while a cluster served, and showed a working model as stopped.
type Cluster interface {
	Health() service.Health
	LastLogs(n int) string
	LogHint() string
}

// clusterControlReason is why the controls are off while a cluster serves.
const clusterControlReason = "A recipe cluster is serving this profile. The panel shows its status " +
	"but can't control it yet: use 'llm-helper restart' or 'down' on the command line."

// SetCluster tells the panel how to find the recipe cluster serving an
// instance. fn returns nil when the instance's own unit serves it.
func (s *Server) SetCluster(fn func(name string) Cluster) { s.clusterFor = fn }

func (s *Server) cluster(name string) Cluster {
	if s.clusterFor == nil {
		return nil
	}
	return s.clusterFor(name)
}

// checkPrivilege works out up front whether restart/stop can succeed, so the
// UI can say why rather than surfacing a bare exit status after the click.
func checkPrivilege(cfg *config.Config) (bool, string) {
	if cfg.ServiceScope() != "system" {
		return true, ""
	}
	if os.Geteuid() == 0 {
		return true, ""
	}
	if runtime.GOOS != "linux" {
		// launchd's system domain is root's, and there is no per-command
		// rule to probe for the way there is with sudoers.
		return false, "The model services run in system scope, which belongs to root. " +
			"Start the panel with sudo, or run the services in user scope."
	}
	// Ask sudo whether the exact commands are permitted rather than whether
	// sudo works at all: a rule granting only these verbs makes `sudo -n
	// true` fail. Every verb the panel uses is checked, against a sample
	// instance unit, so a wildcard rule is what passes.
	systemctl, err := exec.LookPath("systemctl")
	if err != nil {
		systemctl = "/usr/bin/systemctl"
	}
	unit := service.InstanceName("probe")
	allowed := true
	for _, verb := range [][]string{{"restart"}, {"enable"}, {"disable", "--now"}} {
		args := append(append([]string{"-n", "-l", systemctl}, verb...), unit)
		if err := run("sudo", args...); err != nil {
			allowed = false
			break
		}
	}
	if allowed {
		return true, ""
	}
	return false, "The model services run in system scope, and this user can't run systemctl without a password. " +
		"Start the server with sudo, or allow it for the model units:\n  " + sudoersRule(currentUser())
}

// sudoersRule grants the panel's verbs on instance units. In sudoers a * also
// matches spaces, so the negated entry refuses any second argument.
func sudoersRule(user string) string {
	unit := service.InstanceName("*")
	return fmt.Sprintf("%s ALL=(root) NOPASSWD: /usr/bin/systemctl restart %s, /usr/bin/systemctl enable %s, "+
		"/usr/bin/systemctl disable --now %s, !/usr/bin/systemctl * llm-helper@* *",
		user, unit, unit, unit)
}

// Handler builds the routes.
func (s *Server) Handler(static fs.FS) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/status", s.guard(s.handleStatus))
	mux.HandleFunc("GET /api/profiles", s.guard(s.handleProfiles))
	mux.HandleFunc("GET /api/logs", s.guard(s.handleLogs))
	mux.HandleFunc("GET /api/machine", s.guard(s.handleMachine))
	mux.HandleFunc("GET /api/token", s.guard(s.handleToken))
	mux.HandleFunc("GET /api/search", s.guard(s.handleSearch))
	mux.HandleFunc("GET /api/inspect", s.guard(s.handleInspect))
	mux.HandleFunc("GET /api/gguf", s.guard(s.handleGGUF))
	mux.HandleFunc("POST /api/gguf/install", s.guard(s.handleInstallGGUF))
	mux.HandleFunc("POST /api/pull", s.guard(s.handlePull))
	mux.HandleFunc("POST /api/profiles/{name}/remove", s.guard(s.handleRemoveProfile))
	mux.HandleFunc("GET /api/instances", s.guard(s.handleInstances))
	mux.HandleFunc("POST /api/instances/{name}/up", s.guard(s.handleUp))
	mux.HandleFunc("POST /api/instances/{name}/down", s.guard(s.handleDown))
	mux.HandleFunc("POST /api/instances/{name}/restart", s.guard(s.handleRestart))
	mux.HandleFunc("POST /api/instances/{name}/default", s.guard(s.handleDefault))

	// Real pages rather than one document that hides half of itself:
	// watching a machine and choosing a model are separate jobs, and each
	// then loads only its own script. Extensionless so the address bar reads
	// as a place rather than a file.
	mux.Handle("GET /models", s.guard(func(w http.ResponseWriter, r *http.Request) {
		http.ServeFileFS(w, r, static, "models.html")
	}))

	// The docs the binary ships with, rendered from the manifest at first
	// use. Same guard as everything else: they describe how to control this
	// machine, so they are served by the same token.
	docsHandler := s.guard(s.handleDocs)
	mux.HandleFunc("GET /docs", docsHandler)
	mux.HandleFunc("GET /docs/", docsHandler)

	files := http.FileServerFS(static)
	mux.Handle("GET /", s.guard(func(w http.ResponseWriter, r *http.Request) {
		files.ServeHTTP(w, r)
	}))
	return mux
}

// guard authenticates every request.
//
// The panel can stop a running server and rewrite profiles, so it is not
// something to leave open on a tailnet or a LAN. A token is required even on
// loopback: "it's only localhost" stops being true the moment someone adds
// --listen, and a default that is safe only in one configuration is not a
// safe default.
func (s *Server) guard(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !s.authorized(r) {
			// No WWW-Authenticate: a browser password box can't help here,
			// and the URL printed at startup already carries the token.
			denied(w, r, http.StatusUnauthorized, "unauthorized -- open the URL printed by 'llm-helper serve'")
			return
		}
		// Anything that changes state must carry the token in a header.
		// A cookie alone would let any page on the internet POST here with
		// the browser attaching the cookie; a cross-origin script cannot
		// read the token, so it cannot set the header.
		if r.Method != http.MethodGet && !tokenEqual(r.Header.Get(tokenHeader), s.token) {
			denied(w, r, http.StatusForbidden, "this action needs the token in the "+tokenHeader+" header")
			return
		}
		next(w, r)
	}
}

// denied refuses a request in the shape its caller can read. Under /api the
// page's fetch helper reads {"error": ...} and shows it; a plain-text body
// there is discarded and the user is left with "401 Unauthorized" instead of
// the sentence that says what to do. A document request has no such helper,
// so it keeps the plain-text body a browser will render.
func denied(w http.ResponseWriter, r *http.Request, code int, msg string) {
	if strings.HasPrefix(r.URL.Path, "/api/") {
		writeJSON(w, code, map[string]string{"error": msg})
		return
	}
	http.Error(w, msg, code)
}

const (
	tokenHeader = "X-VLLM-Helper-Token"
	tokenCookie = "vllm_helper_token"
)

func (s *Server) authorized(r *http.Request) bool {
	if tokenEqual(r.Header.Get(tokenHeader), s.token) {
		return true
	}
	if c, err := r.Cookie(tokenCookie); err == nil && tokenEqual(c.Value, s.token) {
		return true
	}
	return tokenEqual(r.URL.Query().Get("token"), s.token)
}

func tokenEqual(got, want string) bool {
	return subtle.ConstantTimeCompare([]byte(got), []byte(want)) == 1
}

// SetCookie turns a token in the URL into a session cookie, so the token
// isn't dragged through every subsequent request and the address bar.
func (s *Server) SetCookie(w http.ResponseWriter, r *http.Request) bool {
	if r.URL.Query().Get("token") == "" || !tokenEqual(r.URL.Query().Get("token"), s.token) {
		return false
	}
	http.SetCookie(w, &http.Cookie{
		Name:     tokenCookie,
		Value:    s.token,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteStrictMode,
		MaxAge:   int((30 * 24 * time.Hour).Seconds()),
	})
	return true
}

// handleToken hands the token back to a caller that already proved it has it.
//
// This is what makes a reload work. The cookie is HttpOnly, so the page cannot
// read the token out of it, and without this a reloaded tab could display the
// machine but not change anything -- it had no way to fill in the header that
// mutations require.
//
// It does not weaken that requirement. Reaching this endpoint needs the cookie,
// which is SameSite=Strict and so is not sent on a cross-site request at all;
// and even if it were, the same-origin policy stops another origin from reading
// the response. So the header remains something only a page served from here
// can set, which is the property the mutation check depends on.
func (s *Server) handleToken(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"token": s.token})
}

// Token is the shared secret for this machine.
func (s *Server) Token() string { return s.token }

// loadOrCreateToken keeps the token in the state dir rather than
// secrets.env, because everything in secrets.env is inlined into the
// generated instance env and handed to vLLM's environment. A credential for
// this panel has no business being in the model server's environment.
func loadOrCreateToken(stateDir string) (string, error) {
	path := filepath.Join(stateDir, "web-token")
	if b, err := os.ReadFile(path); err == nil {
		if t := strings.TrimSpace(string(b)); t != "" {
			return t, nil
		}
	}
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	token := hex.EncodeToString(buf)
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(token+"\n"), 0o600); err != nil {
		return "", fmt.Errorf("saving the panel token: %w", err)
	}
	return token, nil
}

// ListenURL renders the address to open, with the token attached.
func ListenURL(addr, token string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		host, port = "localhost", "8080"
	}
	// 0.0.0.0 is where it listens, not somewhere to browse to.
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "localhost"
	}
	return fmt.Sprintf("http://%s/?token=%s", net.JoinHostPort(host, port), token)
}

// LoopbackOnly reports whether an address is reachable only from this
// machine, which decides how loudly to warn about exposure.
func LoopbackOnly(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
