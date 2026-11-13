package cli

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/coder/llm-helper/internal/hf"
)

// A llama.cpp profile's MODEL is <repo>:<quant>, and inspect sent all of it
// to the Hub as the repo id, which answered with a 401. The repo alone is
// looked up.
func TestInspectLooksUpTheRepoOfARepoQuant(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		http.NotFound(w, r)
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	err := runInspect(context.Background(), []string{"unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF:UD-IQ3_XXS"}, true)
	if err == nil {
		t.Fatal("inspect succeeded against a Hub that knows no model")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) == 0 || paths[0] != "/api/models/unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF" {
		t.Errorf("asked the Hub for %q, want the repo without its quant", paths)
	}
}

// A quant the repo does not hold is named as such, with the ones it does.
func TestInspectRefusesAQuantTheRepoDoesNotHold(t *testing.T) {
	t.Setenv("VLLM_HELPER_CONFIG_DIR", t.TempDir())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/org/m-GGUF":
			fmt.Fprint(w, `{"id":"org/m-GGUF","tags":["gguf"]}`)
		case "/api/models/org/m-GGUF/tree/main":
			fmt.Fprint(w, `[{"type":"file","path":"m-Q4_K_M.gguf","size":4000000000},{"type":"file","path":"m-Q8_0.gguf","size":8000000000}]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	defer hf.SetBaseURLForTest(srv.URL)()

	err := runInspect(context.Background(), []string{"org/m-GGUF:Q5_K_M"}, true)
	if err == nil || !strings.Contains(err.Error(), `org/m-GGUF has no quant called "Q5_K_M"`) || !strings.Contains(err.Error(), "Q8_0") {
		t.Errorf("err = %v, want the missing quant and the ones there are", err)
	}
}
