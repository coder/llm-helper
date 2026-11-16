package hf

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const rev = "0123456789abcdef0123456789abcdef01234567"

// cachedRepo writes a repo into a hub cache the way huggingface_hub does,
// with each snapshot file a link to a blob.
func cachedRepo(t *testing.T, hub, repoID string, files map[string]string) {
	t.Helper()
	root := filepath.Join(hub, "models--"+strings.ReplaceAll(repoID, "/", "--"))
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	must(os.MkdirAll(filepath.Join(root, "refs"), 0o755))
	must(os.WriteFile(filepath.Join(root, "refs", "main"), []byte(rev), 0o644))
	i := 0
	for name, body := range files {
		i++
		blob := filepath.Join(root, "blobs", strings.Repeat(string(rune('a'+i)), 40))
		must(os.MkdirAll(filepath.Dir(blob), 0o755))
		must(os.WriteFile(blob, []byte(body), 0o644))
		link := filepath.Join(root, "snapshots", rev, filepath.FromSlash(name))
		must(os.MkdirAll(filepath.Dir(link), 0o755))
		rel, err := filepath.Rel(filepath.Dir(link), blob)
		must(err)
		must(os.Symlink(rel, link))
	}
}

func useCache(t *testing.T, hub string, offline bool) {
	t.Helper()
	UseCache(hub, offline)
	t.Cleanup(func() { UseCache("", false) })
}

// deadHub points the client at a server that is already closed.
func deadHub(t *testing.T) {
	t.Helper()
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()
	old := baseURL
	baseURL = url
	t.Cleanup(func() { baseURL = old })
}

func TestOfflineReadsTheCachedSnapshot(t *testing.T) {
	hub := t.TempDir()
	cachedRepo(t, hub, repo, map[string]string{
		"config.json":           `{"model_type":"qwen3","max_position_embeddings":40960}`,
		"chat_template.jinja":   "{{ messages }}",
		"tokenizer_config.json": `{"chat_template":"x"}`,
		"model.safetensors":     strings.Repeat("w", 64),
	})
	newHub(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("offline asked the Hub for %s", r.URL)
	}))
	useCache(t, hub, true)
	ctx := context.Background()

	if !Offline() {
		t.Fatal("Offline() is false")
	}
	arch, err := Config(ctx, repo)
	if err != nil || arch.ModelType != "qwen3" {
		t.Fatalf("Config = %+v, %v", arch, err)
	}
	if n, err := ContextLimit(ctx, repo); err != nil || n != 40960 {
		t.Fatalf("ContextLimit = %d, %v", n, err)
	}
	if text, found, err := ChatTemplate(ctx, repo); err != nil || !found || text != "{{ messages }}" {
		t.Fatalf("ChatTemplate = %q, %v, %v", text, found, err)
	}
	if m, err := Lookup(ctx, repo); err != nil || m.ID != repo {
		t.Fatalf("Lookup = %+v, %v", m, err)
	}
	tree, err := repoTree(ctx, repo)
	if err != nil || len(tree) != 4 {
		t.Fatalf("repoTree = %+v, %v", tree, err)
	}
	found, err := Search(ctx, "qwen3 fp8", 10)
	if err != nil || len(found) != 1 || found[0].ID != repo {
		t.Fatalf("Search = %+v, %v", found, err)
	}
	if found, _ := Search(ctx, "llama", 10); len(found) != 0 {
		t.Fatalf("Search matched an uncached model: %+v", found)
	}
}

func TestOfflineFileMissingFromACachedRepoIsNotFound(t *testing.T) {
	hub := t.TempDir()
	cachedRepo(t, hub, repo, map[string]string{"model.safetensors": "w"})
	useCache(t, hub, true)

	arch, err := Config(context.Background(), repo)
	if err != nil || arch.ModelType != "" {
		t.Fatalf("Config = %+v, %v; want an empty Arch like a repo with no config.json", arch, err)
	}
	if _, found, err := ChatTemplate(context.Background(), repo); err != nil || found {
		t.Fatalf("ChatTemplate found=%v err=%v", found, err)
	}
}

func TestOfflineUncachedRepoSaysSo(t *testing.T) {
	useCache(t, t.TempDir(), true)
	ctx := context.Background()
	if _, err := Config(ctx, repo); !errors.Is(err, ErrOffline) {
		t.Fatalf("Config err = %v", err)
	}
	if _, err := Lookup(ctx, repo); !errors.Is(err, ErrOffline) {
		t.Fatalf("Lookup err = %v", err)
	}
	if _, err := repoTree(ctx, repo); !errors.Is(err, ErrOffline) {
		t.Fatalf("repoTree err = %v", err)
	}
}

func TestUnreachableHubFallsBackToTheCache(t *testing.T) {
	hub := t.TempDir()
	cachedRepo(t, hub, repo, map[string]string{"config.json": `{"model_type":"qwen3"}`})
	deadHub(t)
	useCache(t, hub, false)

	if Offline() {
		t.Fatal("Offline() is true before any request failed")
	}
	arch, err := Config(context.Background(), repo)
	if err != nil || arch.ModelType != "qwen3" {
		t.Fatalf("Config = %+v, %v", arch, err)
	}
	if !Offline() {
		t.Fatal("Offline() is false after the Hub did not answer")
	}
}

func TestHubIsUsedWhenItAnswers(t *testing.T) {
	hub := t.TempDir()
	cachedRepo(t, hub, repo, map[string]string{"config.json": `{"model_type":"cached"}`})
	newHub(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model_type":"hub"}`))
	}))
	useCache(t, hub, false)

	arch, err := Config(context.Background(), repo)
	if err != nil || arch.ModelType != "hub" {
		t.Fatalf("Config = %+v, %v", arch, err)
	}
	if Offline() {
		t.Fatal("Offline() is true while the Hub answers")
	}
}

func TestNoCacheMeansNoFallback(t *testing.T) {
	deadHub(t)
	UseCache("", false)
	if _, err := Config(context.Background(), repo); err == nil || errors.Is(err, ErrOffline) {
		t.Fatalf("Config err = %v; want the transport error", err)
	}
}

func TestACancelledRequestDoesNotMarkTheHubUnreachable(t *testing.T) {
	hub := t.TempDir()
	deadHub(t)
	useCache(t, hub, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _ = Config(ctx, repo)
	if Offline() {
		t.Fatal("a cancelled request marked the Hub unreachable")
	}
}

func TestHubIsTriedAgainAfterAWhile(t *testing.T) {
	hub := t.TempDir()
	cachedRepo(t, hub, repo, map[string]string{"config.json": `{"model_type":"cached"}`})
	newHub(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model_type":"hub"}`))
	}))
	useCache(t, hub, false)
	cache.Lock()
	cache.downUntil = time.Now().Add(-time.Second)
	cache.Unlock()

	arch, err := Config(context.Background(), repo)
	if err != nil || arch.ModelType != "hub" {
		t.Fatalf("Config = %+v, %v", arch, err)
	}
}

func TestAnEmptyCacheIsNoFallback(t *testing.T) {
	deadHub(t)
	useCache(t, t.TempDir(), false)
	if _, err := Config(context.Background(), repo); err == nil || errors.Is(err, ErrOffline) {
		t.Fatalf("Config err = %v; want the transport error", err)
	}
	if Offline() {
		t.Fatal("Offline() is true with nothing to read")
	}
}
