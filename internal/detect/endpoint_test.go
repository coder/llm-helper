package detect_test

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/detect"
)

// serveOn starts an OpenAI-shaped server and returns the port it listens on.
// apiKey "" means the server accepts anyone.
func serveOn(t *testing.T, apiKey string, health int) string {
	t.Helper()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(health)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		if apiKey != "" && r.Header.Get("Authorization") != "Bearer "+apiKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"org/model-a"},{"id":"org/model-b"}]}`))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return portOf(t, srv.URL)
}

func portOf(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	return u.Port()
}

// closedPort returns a port nothing is listening on.
func closedPort(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	port := portOf(t, srv.URL)
	srv.Close()
	return port
}

func TestProbePortsHealthyServer(t *testing.T) {
	t.Parallel()

	port := serveOn(t, "", http.StatusOK)
	got := detect.ProbePorts([]string{port}, "")
	if len(got) != 1 {
		t.Fatalf("ProbePorts = %#v, want one endpoint", got)
	}
	ep := got[0]
	if ep.Port != port {
		t.Errorf("Port = %q, want %q", ep.Port, port)
	}
	if !ep.Healthy {
		t.Error("Healthy should be true when /health answers 200")
	}
	if ep.NeedsAuth {
		t.Error("NeedsAuth should be false for an open server")
	}
	want := []string{"org/model-a", "org/model-b"}
	if !reflect.DeepEqual(ep.Models, want) {
		t.Errorf("Models = %#v, want %#v", ep.Models, want)
	}
}

func TestProbePortsWrongAPIKey(t *testing.T) {
	t.Parallel()

	port := serveOn(t, "sk-right", http.StatusNotFound)

	t.Run("no key", func(t *testing.T) {
		t.Parallel()
		got := detect.ProbePorts([]string{port}, "")
		if len(got) != 1 {
			t.Fatalf("ProbePorts = %#v, want the endpoint reported even behind auth", got)
		}
		if !got[0].NeedsAuth {
			t.Error("NeedsAuth should be true when /v1/models answers 401")
		}
		if len(got[0].Models) != 0 {
			t.Errorf("Models = %#v, want none without a usable key", got[0].Models)
		}
	})

	t.Run("wrong key", func(t *testing.T) {
		t.Parallel()
		got := detect.ProbePorts([]string{port}, "sk-wrong")
		if len(got) != 1 || !got[0].NeedsAuth {
			t.Errorf("ProbePorts = %#v, want NeedsAuth for a rejected key", got)
		}
	})

	t.Run("right key", func(t *testing.T) {
		t.Parallel()
		got := detect.ProbePorts([]string{port}, "sk-right")
		if len(got) != 1 {
			t.Fatalf("ProbePorts = %#v, want one endpoint", got)
		}
		if got[0].NeedsAuth {
			t.Error("NeedsAuth should be false once the key is accepted")
		}
		if len(got[0].Models) != 2 {
			t.Errorf("Models = %#v, want the served models", got[0].Models)
		}
	})
}

func TestProbePortsIgnoresNothingListening(t *testing.T) {
	t.Parallel()

	if got := detect.ProbePorts([]string{closedPort(t)}, ""); len(got) != 0 {
		t.Errorf("ProbePorts = %#v, want nothing for a closed port", got)
	}
}

// A server that answers HTTP but is not OpenAI-compatible is only worth
// reporting when its health route says so.
func TestProbePortsNonOpenAIServer(t *testing.T) {
	t.Parallel()

	t.Run("unhealthy and not openai", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.NotFoundHandler())
		t.Cleanup(srv.Close)
		if got := detect.ProbePorts([]string{portOf(t, srv.URL)}, ""); len(got) != 0 {
			t.Errorf("ProbePorts = %#v, want nothing", got)
		}
	})

	t.Run("healthy but not openai", func(t *testing.T) {
		t.Parallel()
		mux := http.NewServeMux()
		mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {})
		srv := httptest.NewServer(mux)
		t.Cleanup(srv.Close)
		got := detect.ProbePorts([]string{portOf(t, srv.URL)}, "")
		if len(got) != 1 || !got[0].Healthy || len(got[0].Models) != 0 {
			t.Errorf("ProbePorts = %#v, want a healthy endpoint with no models", got)
		}
	})
}

// A body that isn't the expected JSON must not take the probe down with it.
func TestProbePortsMalformedModelList(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("not json at all"))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	got := detect.ProbePorts([]string{portOf(t, srv.URL)}, "")
	if len(got) != 1 {
		t.Fatalf("ProbePorts = %#v, want the endpoint reported anyway", got)
	}
	if len(got[0].Models) != 0 {
		t.Errorf("Models = %#v, want none from an undecodable body", got[0].Models)
	}
}

// A cluster's head can be another machine, so the watchdog probes it by URL.
// The answer has to match what probing the same server by port says.
func TestProbeURLReachesANamedHost(t *testing.T) {
	t.Parallel()

	port := serveOn(t, "sk-right", http.StatusOK)
	ep, ok := detect.ProbeURL("http://127.0.0.1:"+port+"/", "sk-right")
	if !ok || !ep.Healthy || ep.Port != port || len(ep.Models) != 2 {
		t.Errorf("ProbeURL = %#v, %v; want a healthy endpoint on %s with its models", ep, ok, port)
	}
	if ep, ok := detect.ProbeURL("http://127.0.0.1:"+closedPort(t), ""); ok || ep.Healthy {
		t.Errorf("ProbeURL = %#v, %v; want nothing for a closed port", ep, ok)
	}
}

func TestProbePortsScansEveryPort(t *testing.T) {
	t.Parallel()

	first := serveOn(t, "", http.StatusOK)
	second := serveOn(t, "", http.StatusOK)
	got := detect.ProbePorts([]string{first, closedPort(t), second}, "")
	if len(got) != 2 {
		t.Fatalf("ProbePorts = %#v, want both live ports", got)
	}
	if got[0].Port != first || got[1].Port != second {
		t.Errorf("ports = %q, %q, want %q, %q in the order given", got[0].Port, got[1].Port, first, second)
	}
}

func TestDefaultPorts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		extra []string
		want  []string
	}{{
		name: "no extras",
		want: []string{"8000", "8001", "8080", "8888", "5000"},
	}, {
		name:  "a configured port comes first",
		extra: []string{"9001"},
		want:  []string{"9001", "8000", "8001", "8080", "8888", "5000"},
	}, {
		name:  "a port already in the list isn't repeated",
		extra: []string{"8001"},
		want:  []string{"8001", "8000", "8080", "8888", "5000"},
	}, {
		name:  "empty extras are dropped",
		extra: []string{""},
		want:  []string{"8000", "8001", "8080", "8888", "5000"},
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := detect.DefaultPorts(tt.extra...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("DefaultPorts(%q) = %v, want %v", strings.Join(tt.extra, ","), got, tt.want)
			}
		})
	}
}
