package web

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/control"
	"github.com/coder/vllm-helper/internal/service"
)

// newTestServer builds a server over a throwaway config tree that contains
// secrets, so the leak tests below have something real to find.
func newTestServer(t *testing.T) *Server {
	t.Helper()
	root := t.TempDir()
	dirs := config.Dirs{
		Config:   filepath.Join(root, "config"),
		State:    filepath.Join(root, "state"),
		Data:     filepath.Join(root, "data"),
		Profiles: filepath.Join(root, "config", "models"),
	}
	for _, d := range []string{dirs.Config, dirs.State, dirs.Profiles} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(path, body string) {
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(dirs.ConfigFile(), "PORT=8001\nSERVICE_NAME=vllm.service\nSERVICE_SCOPE=user\n")
	write(dirs.SecretsFile(), "VLLM_API_KEY=sk-do-not-leak-me\nCODER_API_TOKEN=coder-do-not-leak\nHF_TOKEN=hf_do_not_leak\n")
	write(dirs.ProfileFile("demo"), "MODEL=org/demo\nMAX_MODEL_LEN=4096\n")

	cfgFile, err := config.ParseEnvFile(dirs.ConfigFile())
	if err != nil {
		t.Fatal(err)
	}
	secrets, err := config.ParseEnvFile(dirs.SecretsFile())
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{Dirs: dirs, File: cfgFile, Secrets: secrets}

	token, err := loadOrCreateToken(dirs.State)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{
		cfg:        cfg,
		build:      "test",
		token:      token,
		svc:        service.Service{Name: "vllm.service", Scope: "user", NonInteractive: true},
		privileged: true,
	}
	return s
}

// The panel can stop a server and rewrite profiles, so an unauthenticated
// request must not reach a handler even on loopback.
func TestUnauthenticatedRequestsAreRejected(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(Static())

	for _, path := range []string{"/", "/api/status", "/api/profiles", "/api/machine", "/api/logs"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("GET %s without a token = %d, want 401", path, rec.Code)
		}
	}
}

func TestTokenIsAcceptedInHeaderCookieAndQuery(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(Static())

	header := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	header.Header.Set(tokenHeader, s.token)

	cookie := httptest.NewRequest(http.MethodGet, "/api/status", nil)
	cookie.AddCookie(&http.Cookie{Name: tokenCookie, Value: s.token})

	query := httptest.NewRequest(http.MethodGet, "/api/status?token="+s.token, nil)

	for name, r := range map[string]*http.Request{"header": header, "cookie": cookie, "query": query} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		if rec.Code != http.StatusOK {
			t.Errorf("%s auth = %d, want 200", name, rec.Code)
		}
	}

	wrong := httptest.NewRequest(http.MethodGet, "/api/status?token=nope", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, wrong)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("a wrong token = %d, want 401", rec.Code)
	}
}

// A cookie alone must not be enough to change anything. Any page on the
// internet can POST here and the browser would attach the cookie; it cannot
// read the token, so requiring it in a header is what stops the request.
func TestMutationsNeedTheTokenInAHeader(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(Static())

	cookieOnly := httptest.NewRequest(http.MethodPost, "/api/restart", nil)
	cookieOnly.AddCookie(&http.Cookie{Name: tokenCookie, Value: s.token})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, cookieOnly)
	if rec.Code != http.StatusForbidden {
		t.Errorf("POST with only a cookie = %d, want 403", rec.Code)
	}

	queryOnly := httptest.NewRequest(http.MethodPost, "/api/restart?token="+s.token, nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, queryOnly)
	if rec.Code != http.StatusForbidden {
		t.Errorf("POST with only a query token = %d, want 403", rec.Code)
	}
}

// The page's fetch helper reads body.error and shows it. A plain-text refusal
// under /api is discarded there, leaving the user with a bare "401
// Unauthorized" instead of the sentence saying where to find the token.
func TestADeniedAPIRequestAnswersInJSON(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(Static())

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/status", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("GET /api/status without a token = %d, want 401", rec.Code)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("the refusal is not JSON, so the page cannot show it: %v (body %q)", err, rec.Body.String())
	}
	if body.Error == "" {
		t.Errorf("refusal carried no error field: %q", rec.Body.String())
	}

	// A document request has no fetch helper, so it keeps the plain-text
	// body a browser will render.
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("GET / without a token = %d, want 401", rec.Code)
	}
	if json.Valid(rec.Body.Bytes()) {
		t.Errorf("a browser was sent JSON to read: %q", rec.Body.String())
	}
}

// Profile names come from the model half of a repo path, so two owners
// publishing the same model name land on one name. Treating that as "already
// have it" would offer to serve weights nobody asked for.
func TestProfileHoldsDoesNotHandOneOwnersProfileToAnother(t *testing.T) {
	s := newTestServer(t) // holds a profile "demo" whose MODEL is org/demo

	if taken, mine := s.profileHolds("demo", "org/demo"); !taken || !mine {
		t.Errorf("profileHolds(demo, org/demo) = %v/%v, want the repo that saved it recognised", taken, mine)
	}
	if taken, mine := s.profileHolds("demo", "other/demo"); !taken || mine {
		t.Errorf("profileHolds(demo, other/demo) = %v/%v, want taken by someone else's weights", taken, mine)
	}
}

// Every response is assembled from an explicit DTO precisely so a credential
// can't ride along. This asserts that end to end, against a config that
// really does hold secrets.
func TestNoResponseLeaksACredential(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(Static())

	secrets := []string{"sk-do-not-leak-me", "coder-do-not-leak", "hf_do_not_leak"}
	for _, path := range []string{"/api/status", "/api/profiles", "/api/machine", "/api/logs"} {
		r := httptest.NewRequest(http.MethodGet, path, nil)
		r.Header.Set(tokenHeader, s.token)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)

		body := rec.Body.String()
		for _, secret := range secrets {
			if strings.Contains(body, secret) {
				t.Errorf("%s leaked a credential: %s", path, body)
			}
		}
	}
}

// The token lives in the state dir rather than secrets.env because
// everything in secrets.env is inlined into active.env and handed to vLLM's
// environment. A panel credential has no business being there.
func TestTokenIsStoredPrivatelyAndReused(t *testing.T) {
	dir := t.TempDir()
	first, err := loadOrCreateToken(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) < 32 {
		t.Errorf("token is %d chars, want something long enough not to guess", len(first))
	}
	again, err := loadOrCreateToken(dir)
	if err != nil {
		t.Fatal(err)
	}
	if again != first {
		t.Error("a second call minted a new token; every restart would invalidate open tabs")
	}
	info, err := os.Stat(filepath.Join(dir, "web-token"))
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("token file mode = %v, want 0600", perm)
	}
}

// Without privilege the buttons are disabled and the reason is shown, so a
// click can't fail with a bare "exit status 1".
func TestActionsRefuseWithoutPrivilege(t *testing.T) {
	s := newTestServer(t)
	s.setPrivilege(false, "needs passwordless sudo")
	h := s.Handler(Static())

	r := httptest.NewRequest(http.MethodPost, "/api/restart", nil)
	r.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)

	if rec.Code != http.StatusForbidden {
		t.Fatalf("restart without privilege = %d, want 403", rec.Code)
	}
	var got actionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got.Error, "sudo") {
		t.Errorf("error = %q, want it to explain what's missing", got.Error)
	}
}

func TestListenURL(t *testing.T) {
	t.Parallel()
	// 0.0.0.0 is where it binds, not somewhere anyone can browse to.
	if got := ListenURL("0.0.0.0:8080", "abc"); got != "http://localhost:8080/?token=abc" {
		t.Errorf("ListenURL(0.0.0.0) = %q", got)
	}
	if got := ListenURL("127.0.0.1:9000", "abc"); got != "http://127.0.0.1:9000/?token=abc" {
		t.Errorf("ListenURL(loopback) = %q", got)
	}
}

func TestLoopbackOnly(t *testing.T) {
	t.Parallel()
	for addr, want := range map[string]bool{
		"127.0.0.1:8080": true,
		"localhost:8080": true,
		"[::1]:8080":     true,
		"0.0.0.0:8080":   false,
		"192.168.1.5:80": false,
	} {
		if got := LoopbackOnly(addr); got != want {
			t.Errorf("LoopbackOnly(%q) = %v, want %v", addr, got, want)
		}
	}
}

// The panel polls, and every health probe is a line in the log of the very
// server someone is likely reading. Several tabs open on one machine must not
// multiply into several streams of them, so a recent answer is reused.
func TestHealthProbeIsReusedBetweenCallers(t *testing.T) {
	var probes int32
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			atomic.AddInt32(&probes, 1)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer vllm.Close()

	s := newTestServer(t)
	s.cfg.File.Set("PORT", portOf(t, vllm.URL))

	// Ten callers, as ten open tabs would be.
	var wg sync.WaitGroup
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, ok := s.health(); !ok {
				t.Error("health() = not ok, want the endpoint reported healthy")
			}
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt32(&probes); got != 1 {
		t.Errorf("probes = %d, want 1: concurrent callers should share one probe", got)
	}

	// And an answer older than the TTL is not reused, or the panel would
	// never notice the server going down.
	s.healthMu.Lock()
	s.healthAt = time.Now().Add(-2 * healthProbeTTL)
	s.healthMu.Unlock()
	if _, ok := s.health(); !ok {
		t.Error("health() = not ok after expiry")
	}
	if got := atomic.LoadInt32(&probes); got != 2 {
		t.Errorf("probes = %d, want 2: a stale answer should be replaced", got)
	}
}

// portOf pulls the port out of an httptest URL.
func portOf(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	return u.Port()
}

// Reloading the page drops the token from the URL -- it is stripped from the
// address bar on the first visit -- and the cookie is HttpOnly, so the page
// cannot read it back. A tab in that state must still be able to act, which
// means recovering the token from the server over the cookie it does have.
func TestReloadedTabCanRecoverTheTokenAndMutate(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	// A tab with only the cookie, exactly as it is after a reload.
	cookie := &http.Cookie{Name: tokenCookie, Value: s.token}

	req := httptest.NewRequest("GET", "/api/token", nil)
	req.AddCookie(cookie)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/token = %d, want 200: a reloaded tab cannot recover the token", rec.Code)
	}
	var body struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Token != s.token {
		t.Fatal("GET /api/token returned the wrong token")
	}

	// The recovered token must satisfy the mutation check -- the actual
	// symptom was a restart button that refused with "no token".
	req = httptest.NewRequest("POST", "/api/restart", nil)
	req.AddCookie(cookie)
	req.Header.Set(tokenHeader, body.Token)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code == http.StatusForbidden || rec.Code == http.StatusUnauthorized {
		t.Errorf("POST /api/restart = %d, want the recovered token to be accepted", rec.Code)
	}

	// And the endpoint is not a way in: without the cookie it says nothing.
	req = httptest.NewRequest("GET", "/api/token", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated GET /api/token = %d, want 401", rec.Code)
	}
	if strings.Contains(rec.Body.String(), s.token) {
		t.Error("unauthenticated GET /api/token leaked the token")
	}
}

// The model-management endpoints reach the network and write profiles, so
// they must sit behind the same guard as everything else. An unauthenticated
// caller reaching any of them would be able to make this machine fetch from
// the internet and rewrite what it serves.
func TestModelEndpointsRequireAuth(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	for _, c := range []struct{ method, path string }{
		{"GET", "/api/search?q=qwen"},
		{"GET", "/api/inspect?repo=Qwen/Qwen3-0.6B"},
		{"POST", "/api/pull?repo=Qwen/Qwen3-0.6B"},
		{"POST", "/api/profiles/demo/remove"},
	} {
		req := httptest.NewRequest(c.method, c.path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s %s = %d, want 401", c.method, c.path, rec.Code)
		}
	}

	// And the mutations need the token in a header, not just the cookie:
	// a cookie alone would let another site POST here.
	for _, c := range []struct{ method, path string }{
		{"POST", "/api/pull?repo=Qwen/Qwen3-0.6B"},
		{"POST", "/api/profiles/demo/remove"},
	} {
		req := httptest.NewRequest(c.method, c.path, nil)
		req.AddCookie(&http.Cookie{Name: tokenCookie, Value: s.token})
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Errorf("%s %s with cookie only = %d, want 403", c.method, c.path, rec.Code)
		}
	}
}

// A repo id off a query string reaches both an outbound URL and, via
// NameFromRepo, a filename. It must be rejected before either, and rejected
// without the handler having gone to the network to find out.
func TestMalformedRepoIsRejectedBeforeAnyWork(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	for _, repo := range []string{
		"", "../../etc/passwd", "owner/..", "noslash", "owner/na me",
		"https://evil.test/x", "a/b/c",
	} {
		path := "/api/inspect?repo=" + url.QueryEscape(repo)
		req := httptest.NewRequest("GET", path, nil)
		req.Header.Set(tokenHeader, s.token)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("GET /api/inspect?repo=%q = %d, want 400", repo, rec.Code)
		}
	}

	// The same guard has to hold on the write path, which is the one that
	// would create a file.
	for _, repo := range []string{"owner/..", "../../etc/passwd", "a/b/c"} {
		path := "/api/pull?repo=" + url.QueryEscape(repo)
		req := httptest.NewRequest("POST", path, nil)
		req.Header.Set(tokenHeader, s.token)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("POST /api/pull?repo=%q = %d, want 400", repo, rec.Code)
		}
	}
}

// Removing the model currently being served would leave the running service
// with no profile behind it, so it is refused rather than allowed to happen
// and be discovered at the next restart.
func TestRemoveRefusesTheActiveProfile(t *testing.T) {
	s := newTestServer(t)
	// The active profile is state, not config: it lives in its own file so a
	// switch does not have to rewrite config.env.
	if err := os.WriteFile(s.cfg.Dirs.ActiveProfileFile(), []byte("demo\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	h := s.Handler(nil)

	req := httptest.NewRequest("POST", "/api/profiles/demo/remove", nil)
	req.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Errorf("removing the active profile = %d, want 409", rec.Code)
	}
	if _, err := os.Stat(s.cfg.Dirs.ProfileFile("demo")); err != nil {
		t.Error("the active profile was deleted despite the refusal")
	}

	// A profile that does not exist is a 404, not a 500 and not a stack
	// trace -- the name comes off a URL.
	req = httptest.NewRequest("POST", "/api/profiles/nope/remove", nil)
	req.Header.Set(tokenHeader, s.token)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Errorf("removing a missing profile = %d, want 404", rec.Code)
	}
}

// A quant name off a query string decides what the machine serves, so it is
// resolved against the repo rather than trusted. Anything the repo does not
// hold has to be refused before the pull, because pulling restarts the service
// and a model that cannot start takes the running one down with it.
//
// Deliberately uses a repo id that cannot resolve rather than a real one: the
// point is that an unresolvable choice never reaches the profile writer, and
// a test that depends on the Hub answering would be asserting the Hub.
func TestPullRefusesAQuantItCannotResolve(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	req := httptest.NewRequest("POST",
		"/api/pull?repo=vllm-helper-test/definitely-not-a-real-repo&quant="+
			url.QueryEscape("Q8_0"), nil)
	req.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code == http.StatusOK {
		t.Errorf("POST /api/pull with an unresolvable quant = 200, want a refusal")
	}
	if strings.Contains(rec.Body.String(), "sk-do-not-leak-me") {
		t.Error("the refusal leaked a credential")
	}
	entries, err := os.ReadDir(s.cfg.Dirs.Profiles)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Name() != "demo.env" {
			t.Errorf("a profile was written for a refused pull: %s", e.Name())
		}
	}
}

// The panel offers the same install the CLI does, so the endpoints behind it
// need the same guard as every other mutation -- this one runs pip against the
// machine's vLLM environment.
func TestGGUFEndpointsAreGuarded(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	for _, c := range []struct{ method, path string }{
		{"GET", "/api/gguf"},
		{"POST", "/api/gguf/install"},
	} {
		req := httptest.NewRequest(c.method, c.path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s %s = %d, want 401", c.method, c.path, rec.Code)
		}
	}

	// Installing is a mutation, so a cookie alone must not be enough.
	req := httptest.NewRequest("POST", "/api/gguf/install", nil)
	req.AddCookie(&http.Cookie{Name: tokenCookie, Value: s.token})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Errorf("POST /api/gguf/install with cookie only = %d, want 403", rec.Code)
	}
}

// The panel shows the command and the caveats rather than hardcoding them, so
// what a browser is told cannot drift from what the CLI says.
func TestGGUFStatusReportsTheSameFactsAsTheCLI(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	req := httptest.NewRequest("GET", "/api/gguf", nil)
	req.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/gguf = %d, want 200", rec.Code)
	}
	var got struct {
		Supported bool     `json:"supported"`
		Command   string   `json:"command"`
		Caveats   []string `json:"caveats"`
		Reason    string   `json:"reason"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Caveats) != len(control.GGUFCaveats()) {
		t.Errorf("caveats = %d, want the %d control publishes",
			len(got.Caveats), len(control.GGUFCaveats()))
	}
	// The test config has no real vLLM, so there is no interpreter to install
	// into -- which must be reported rather than offered as a button.
	if got.Command == "" && got.Reason == "" {
		t.Error("with no interpreter found, one of command or reason must say so")
	}
}

// The plugin builds a CUDA extension from source, which is minutes. A POST
// held open that long is dropped by browsers and proxies, and when it is, the
// page cannot tell a finished install from a lost one -- the symptom being a
// card that never updates until you search for the model again.
//
// So the request must return at once and the work outlive it.
func TestInstallGGUFReturnsWithoutWaiting(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	// The test config points at no real vLLM, so this is refused before any
	// work starts -- which is itself the thing being checked: the handler
	// answers rather than blocking.
	done := make(chan int, 1)
	go func() {
		req := httptest.NewRequest("POST", "/api/gguf/install", nil)
		req.Header.Set(tokenHeader, s.token)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		done <- rec.Code
	}()

	select {
	case code := <-done:
		// Accepted when it started, 400 when there is nowhere to install to.
		// Either is a prompt answer; a hang is what this test exists to catch.
		if code != http.StatusAccepted && code != http.StatusBadRequest {
			t.Errorf("POST /api/gguf/install = %d, want 202 or 400", code)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("POST /api/gguf/install blocked -- it must start the work and return")
	}
}

// Progress lives on the server, so a reloaded page rejoins an install it did
// not start rather than offering to start a second one.
func TestGGUFStatusReportsAnInstallInFlight(t *testing.T) {
	s := newTestServer(t)
	h := s.Handler(nil)

	if !s.beginInstall() {
		t.Fatal("beginInstall refused on a fresh server")
	}
	// A second one must be refused while the first runs: two pips writing to
	// one site-packages is not something to do.
	if s.beginInstall() {
		t.Error("beginInstall allowed a second concurrent install")
	}

	req := httptest.NewRequest("GET", "/api/gguf", nil)
	req.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	var got struct {
		Installing bool   `json:"installing"`
		Error      string `json:"install_error"`
		Log        string `json:"install_log"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Installing {
		t.Error("a running install is not reported, so a reloaded page would offer to start another")
	}

	// And the outcome is readable afterwards, log included, since a failed
	// build explains itself in pip's last lines and a browser has no journal.
	s.finishInstall(errors.New("build failed"), "gcc: fatal error")
	req2 := httptest.NewRequest("GET", "/api/gguf", nil)
	req2.Header.Set(tokenHeader, s.token)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req2)
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Installing {
		t.Error("a finished install is still reported as running")
	}
	if got.Error == "" || got.Log == "" {
		t.Errorf("a failed install must surface its reason and log, got err=%q log=%q",
			got.Error, got.Log)
	}
}

// fakeCluster stands in for a recipe cluster the CLI found serving.
type fakeCluster struct{ health service.Health }

func (f fakeCluster) Health() service.Health { return f.health }
func (f fakeCluster) LastLogs(int) string    { return "cluster log line" }
func (f fakeCluster) LogHint() string        { return "docker logs vllm-node on the head node" }

// While a recipe cluster serves, the native unit is idle. Reading the unit
// reported a working model as stopped, and its buttons would have started a
// second vLLM beside the cluster or claimed a stop that changed nothing.
func TestClusterIsReportedAndNotDriven(t *testing.T) {
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer vllm.Close()

	s := newTestServer(t)
	s.cfg.File.Set("PORT", portOf(t, vllm.URL))
	s.SetCluster(func() Cluster { return fakeCluster{health: service.Health{State: "active", SubState: "running"}} })

	got := s.status()
	if got.Phase != "serving" || !got.Healthy {
		t.Errorf("phase = %q healthy = %v, want serving and healthy: the cluster is up", got.Phase, got.Healthy)
	}
	if got.Service != "cluster" || !got.Cluster || got.ControlReason == "" {
		t.Errorf("service = %q cluster = %v reason = %q, want the cluster named and the controls explained", got.Service, got.Cluster, got.ControlReason)
	}

	h := s.Handler(Static())
	do := func(method, target string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, target, nil)
		r.Header.Set(tokenHeader, s.token)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, r)
		return rec
	}
	for _, target := range []string{"/api/restart", "/api/stop", "/api/switch/demo", "/api/pull?repo=org/other"} {
		rec := do(http.MethodPost, target)
		if rec.Code != http.StatusConflict {
			t.Errorf("POST %s with a cluster serving = %d, want 409", target, rec.Code)
		}
		if !strings.Contains(rec.Body.String(), "command line") {
			t.Errorf("POST %s: body %q should say where to control the cluster", target, rec.Body.String())
		}
	}

	rec := do(http.MethodGet, "/api/logs")
	if !strings.Contains(rec.Body.String(), "cluster log line") {
		t.Errorf("logs = %q, want the cluster's log, not the idle unit's", rec.Body.String())
	}

	// No cluster: the native unit is back in charge and nothing is refused
	// on the cluster's account.
	s.SetCluster(func() Cluster { return nil })
	if got := s.status(); got.Cluster || got.Service != "vllm.service" {
		t.Errorf("without a cluster: cluster = %v service = %q, want the native unit", got.Cluster, got.Service)
	}
}

// A measured budget of 0 is what the models page most needs to say, and
// reported as a bare 0 it read as "no budget" and the page said nothing.
func TestMachineReportsAKnownZeroBudget(t *testing.T) {
	bin := t.TempDir()
	smi := "#!/bin/sh\ncase \"$1\" in\n--query-gpu=index,uuid,name,memory.total) printf '0, GPU-aaaa, NVIDIA RTX A6000, 49140\\n' ;;\n*) exit 1 ;;\nesac\n"
	if err := os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte(smi), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("CUDA_VISIBLE_DEVICES", "")

	s := newTestServer(t)
	r := httptest.NewRequest(http.MethodGet, "/api/machine", nil)
	r.Header.Set(tokenHeader, s.token)
	rec := httptest.NewRecorder()
	s.Handler(Static()).ServeHTTP(rec, r)
	var got struct {
		BudgetBytes int64  `json:"budget_bytes"`
		BudgetKnown bool   `json:"budget_known"`
		Reason      string `json:"budget_reason"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("GET /api/machine = %d %s: %v", rec.Code, rec.Body, err)
	}
	if got.BudgetBytes != 0 || !got.BudgetKnown || !strings.Contains(got.Reason, "CUDA_VISIBLE_DEVICES") {
		t.Errorf("every GPU hidden: %+v, want a known 0 that says why", got)
	}
}
