package recipe

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func deepseek(t *testing.T) TestedRecipe {
	t.Helper()
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	r, ok := tt.Get("deepseek-v4-flash-2spark")
	if !ok {
		t.Fatal("deepseek-v4-flash-2spark missing")
	}
	return r
}

// An unresolved placeholder must fail rather than reach vLLM as an empty
// string, which is how a run silently falls back to the wrong interface.
func TestNodeEnvRefusesUndiscoveredPlaceholders(t *testing.T) {
	r := deepseek(t)
	if _, err := r.NodeEnv(map[string]string{}); err == nil {
		t.Fatal("NodeEnv should fail when nothing was discovered")
	}
	full := map[string]string{
		PlaceholderIface:  "eth9",
		PlaceholderIBIf:   "rocea,roceb",
		PlaceholderNodeIP: "10.0.0.7",
	}
	env, err := r.NodeEnv(full)
	if err != nil {
		t.Fatalf("NodeEnv: %v", err)
	}
	if env["NCCL_SOCKET_IFNAME"] != "eth9" || env["NCCL_IB_HCA"] != "rocea,roceb" || env["VLLM_HOST_IP"] != "10.0.0.7" {
		t.Errorf("placeholders not substituted: %v", env)
	}
	if env["CUTE_DSL_ARCH"] != "sm_121a" {
		t.Errorf("literal value was rewritten: %q", env["CUTE_DSL_ARCH"])
	}
	for _, name := range r.Placeholders() {
		if full[name] == "" {
			t.Errorf("Placeholders() reported %q, which this test does not supply", name)
		}
	}
}

// sparkDiscovered is what discovery reports on the Spark pair, by rank.
func sparkDiscovered(rank int) map[string]string {
	return map[string]string{
		PlaceholderIface:  "enp1s0f1np1",
		PlaceholderIBIf:   "rocep1s0f1,roceP2p1s0f1",
		PlaceholderNodeIP: fmt.Sprintf("10.44.0.%d", rank+1),
	}
}

func rankSpec(t *testing.T, r TestedRecipe, rank int, apiKey string) ContainerSpec {
	t.Helper()
	s, err := r.RankSpec(Rank{
		Rank: rank, MasterAddr: "10.44.0.1", MasterPort: 29501,
		Discovered: sparkDiscovered(rank), HomeDir: "/home/u",
		ServeArgs: r.ServeArgs(), APIKey: apiKey, Container: "vllm-node",
	})
	if err != nil {
		t.Fatalf("%s rank %d: %v", r.ID, rank, err)
	}
	return s
}

// Moving vLLM's container launch into its engine definition must not change
// a byte of what the Sparks run. The golden files were rendered by the
// launcher that had vLLM's command written in Go.
func TestVLLMRecipesRenderTheirGoldenCommands(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatalf("LoadTested: %v", err)
	}
	checked := 0
	for _, r := range tt.Recipes {
		if r.Engine().ID != "vllm" {
			continue
		}
		want, err := os.ReadFile(filepath.Join("testdata", "golden-"+r.ID+".txt"))
		if err != nil {
			t.Fatalf("%s: %v", r.ID, err)
		}
		var got strings.Builder
		for rank := 0; rank < r.Nodes; rank++ {
			line, err := rankSpec(t, r, rank, "sk-test").DisplayCommand()
			if err != nil {
				t.Fatal(err)
			}
			got.WriteString(line + "\n")
		}
		if got.String() != string(want) {
			t.Errorf("%s: command changed\n got %s\nwant %s", r.ID, got.String(), want)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no vLLM recipe was checked against a golden command")
	}
}

// Only rank 0 serves the API; a second non-headless rank would race for the port.
func TestVLLMWorkersRunHeadless(t *testing.T) {
	r := deepseek(t)
	head := strings.Join(rankSpec(t, r, 0, "").Args, " ")
	if strings.Contains(head, "--headless") {
		t.Errorf("rank 0 must serve the API: %s", head)
	}
	if !strings.Contains(head, "--node-rank 0") || !strings.Contains(head, "--master-addr 10.44.0.1") {
		t.Errorf("head args wrong: %s", head)
	}
	worker := strings.Join(rankSpec(t, r, 1, "").Args, " ")
	if !strings.Contains(worker, "--headless") || !strings.Contains(worker, "--node-rank 1") {
		t.Errorf("worker args wrong: %s", worker)
	}
	if got := r.LaunchOrder(); len(got) != 2 || got[0] != 1 || got[1] != 0 {
		t.Errorf("LaunchOrder() = %v; workers must be listening before the head's rendezvous", got)
	}
}

func TestRunArgsRequiresADigest(t *testing.T) {
	r := deepseek(t)
	s := rankSpec(t, r, 0, "")
	s.Image = "eugr/spark-vllm-b12x:latest"
	if _, err := s.RunArgs(); err == nil {
		t.Fatal("a moving tag must be refused; it is how a verified backend name becomes invalid")
	}
	s.Image = r.Image
	args, err := s.RunArgs()
	if err != nil {
		t.Fatalf("RunArgs: %v", err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"--network host", "--ipc host", "--privileged", "--gpus all", "--entrypoint vllm " + r.Image + " serve"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %s", want, joined)
		}
	}
	// A node that dies is diagnosed from its logs, and --rm deletes them.
	if strings.Contains(joined, "--rm") {
		t.Errorf("--rm discards the logs needed to explain a dead node: %s", joined)
	}
}

// The compilation-config value is JSON; losing its quoting corrupts the launch.
func TestShellCommandKeepsJSONIntact(t *testing.T) {
	line, err := rankSpec(t, deepseek(t), 0, "").ShellCommand()
	if err != nil {
		t.Fatalf("ShellCommand: %v", err)
	}
	if !strings.Contains(line, `'{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`) {
		t.Errorf("compilation-config lost its quoting: %s", line)
	}
	if !strings.Contains(line, "--host 0.0.0.0") || !strings.Contains(line, "--port 8001") {
		t.Errorf("host/port not supplied by the launcher: %s", line)
	}
}

// The key must reach the container without appearing on any command line:
// docker's argv names the variable only, and the value comes from an
// assignment the shell consumes before docker starts.
func TestAPIKeyStaysOffDockerArgv(t *testing.T) {
	s := ContainerSpec{
		Name: "n", Image: "img@sha256:abc", Entrypoint: "vllm",
		Args: []string{"serve", "m"}, APIKey: "sk-it's-secret", KeyEnv: "VLLM_API_KEY",
	}
	args, err := s.RunArgs()
	if err != nil {
		t.Fatalf("RunArgs: %v", err)
	}
	for i, a := range args {
		if strings.Contains(a, "sk-it") {
			t.Fatalf("key on docker argv at %d: %q", i, args)
		}
	}
	if !strings.Contains(strings.Join(args, " "), "-e VLLM_API_KEY --entrypoint") {
		t.Errorf("docker is not told to pass VLLM_API_KEY through: %q", args)
	}

	line, err := s.ShellCommand()
	if err != nil {
		t.Fatalf("ShellCommand: %v", err)
	}
	if !strings.HasPrefix(line, `VLLM_API_KEY='sk-it'"'"'s-secret' docker run `) {
		t.Errorf("key is not assigned ahead of docker: %s", line)
	}
	shown, err := s.DisplayCommand()
	if err != nil {
		t.Fatalf("DisplayCommand: %v", err)
	}
	if strings.Contains(shown, "sk-it") || !strings.Contains(shown, "<redacted>") {
		t.Errorf("display command leaks or drops the redaction: %s", shown)
	}

	// A key with nowhere to go must fail rather than be dropped silently.
	s.KeyEnv = ""
	if _, err := s.RunArgs(); err == nil {
		t.Error("a key with no variable to pass it in was accepted")
	}
}

// The GLM recipe is what the Sparks serve through TensorFold, so each rank
// must run exactly the command the serving deployment's containers run: the
// argv below was read from them with docker inspect.
func TestTensorFoldRecipeRendersTheServingCommand(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatal(err)
	}
	r, ok := tt.Get("glm-5.3-flash-tf-2spark")
	if !ok {
		t.Fatal("glm-5.3-flash-tf-2spark missing")
	}
	weights := "/root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c"
	drafter := "/root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"
	shared := []string{"--drafter", drafter, "--context", "1048576", "--parallel", "4", "--max-tokens", "32768", "--thinking", "--vision",
		"--spill-gib", "128", "--snapshot-dir", "/spill", "--spill-highwater", "0.70", "--spill-min-tokens", "8192", "--spill-min-free-gib", "50"}
	want := map[int][]string{
		0: append([]string{"tensorfold", "serve", weights, "--tp", "2", "--rank", "0", "--master", "10.44.0.1", "--master-port", "29551",
			"--name", "GLM-5.3-Flash-EXL3", "--host", "127.0.0.1", "--port", "8889"}, shared...),
		1: append([]string{"tensorfold", "serve", weights, "--tp", "2", "--rank", "1", "--master", "10.44.0.1", "--master-port", "29551"}, shared...),
	}
	for rank, args := range want {
		s, err := r.RankSpec(Rank{
			Rank: rank, MasterAddr: "10.44.0.1", MasterPort: 29551,
			Discovered: sparkDiscovered(rank), HomeDir: "/home/u",
			ServeArgs: r.ServeArgs(), APIKey: "sk-test", Container: "glm53-flash-tf",
		})
		if err != nil {
			t.Fatalf("rank %d: %v", rank, err)
		}
		if got := strings.Join(s.Args, " "); got != strings.Join(args, " ") {
			t.Errorf("rank %d argv\n got %s\nwant %s", rank, got, strings.Join(args, " "))
		}
		// The image's own entrypoint sets up CUDA, so it has to stay.
		if s.Entrypoint != "" {
			t.Errorf("rank %d replaces the image's entrypoint with %q", rank, s.Entrypoint)
		}
		// TensorFold checks no key, so none is passed, on argv or otherwise.
		if s.APIKey != "" || s.KeyEnv != "" {
			t.Errorf("rank %d was handed an API key it does not check", rank)
		}
		line, err := s.ShellCommand()
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(line, "sk-test") || strings.Contains(line, "--privileged") {
			t.Errorf("rank %d: %s", rank, line)
		}
		for _, m := range []string{
			"/home/u/.cache/huggingface:/root/.cache/huggingface",
			"/home/u/.cache/tensorfold-glm53/789e8f3bdfa2:/cache",
			"/home/u/.cache/tensorfold-spill:/spill",
		} {
			if !strings.Contains(line, "-v "+m) {
				t.Errorf("rank %d lacks mount %s: %s", rank, m, line)
			}
		}
		if s.Env["NCCL_SOCKET_IFNAME"] != "enp1s0f1np1" || s.Env["TF_GLM_KEEP_PER_CHAT"] != "2" || s.Env["HF_HUB_OFFLINE"] != "1" {
			t.Errorf("rank %d env: %v", rank, s.Env)
		}
	}
	if got := r.HealthPath(); got != "/v1/models" {
		t.Errorf("HealthPath() = %q, want /v1/models, which answers only once the model is loaded", got)
	}
	if got := r.ContextLimit(); got != "1048576" {
		t.Errorf("ContextLimit() = %q, want the --context the recipe sets", got)
	}
	if got := r.LaunchOrder(); len(got) != 2 || got[0] != 1 {
		t.Errorf("LaunchOrder() = %v; the worker must be listening before rank 0's rendezvous", got)
	}
}

// A launch checks every pinned revision is in each node's cache before it
// stops anything, and names what is missing.
func TestStagingScriptChecksEveryPinnedRevision(t *testing.T) {
	tt, err := LoadTested()
	if err != nil {
		t.Fatal(err)
	}
	r, _ := tt.Get("glm-5.3-flash-tf-2spark")
	script := r.StagingScript()
	for _, m := range r.Models {
		if !strings.Contains(script, "models--"+strings.ReplaceAll(m.ID, "/", "--")+"/snapshots/"+m.Revision) ||
			!strings.Contains(script, "echo "+m.ID+"@"+m.Revision) {
			t.Errorf("script does not check %s@%s:\n%s", m.ID, m.Revision, script)
		}
	}
	home := t.TempDir()
	dir := filepath.Join(home, ".cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("sh", "-c", script)
	cmd.Env = []string{"HOME=" + home, "PATH=" + os.Getenv("PATH")}
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(out)); got != "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@078455ffe6472f9a52fbc1139f58b9db2881b25c" {
		t.Errorf("missing = %q, want only the weights", got)
	}
	// vLLM's recipes pin no models, so they check nothing.
	if s := deepseek(t).StagingScript(); s != "" {
		t.Errorf("a recipe without pinned models got a staging check: %s", s)
	}
}

func TestShellQuote(t *testing.T) {
	cases := map[string]string{
		"plain":       "plain",
		"a=b":         "a=b",
		"rocea,roceb": "rocea,roceb",
		"":            "''",
		`{"a":1}`:     `'{"a":1}'`,
		"has space":   "'has space'",
		"it's":        `'it'"'"'s'`,
	}
	for in, want := range cases {
		if got := ShellQuote(in); got != want {
			t.Errorf("ShellQuote(%q) = %q, want %q", in, got, want)
		}
	}
}
