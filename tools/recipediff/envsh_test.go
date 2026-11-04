package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testdata/miaai-glm53/config.sh is MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-
// Sparks-TensorFold's scripts/config.sh at bf5f9452, unmodified (Apache-2.0),
// the file the shipped TensorFold recipe cites.
const envShFixtureDir = "testdata/miaai-glm53"

func loadEnvSh(t *testing.T) envSh {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(envShFixtureDir, "config.sh"))
	if err != nil {
		t.Fatal(err)
	}
	return parseEnvSh(b)
}

// Only a default anyone could restate is read: a literal, a nested default,
// or another key's. A computed or conditional one is left out, not guessed.
func TestEnvShReadsLiteralDefaults(t *testing.T) {
	s := loadEnvSh(t)
	for key, want := range map[string]string{
		"SPILL_GIB":                  "0",
		"HOST":                       "0.0.0.0",
		"PORT":                       "8888",
		"TF_GLM_EFFORT_TAIL":         "0",
		"TF_GLM_KEEP_PER_CHAT":       "0",
		"TF_GLM_KV":                  "fp8", // export TF_GLM_KV="$KV"
		"TENSORFOLD_NUCLEUS_UNION":   "1",   // ${NUCLEUS_UNION:-${TENSORFOLD_NUCLEUS_UNION:-1}}
		"TF_GLM_STREAM_SMOOTH_MS":    "400", // the second assignment on its line
		"ABLIT_ID":                   "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit",
		"TENSORFOLD_NO_UPDATE_CHECK": "1",
	} {
		if got, ok := s.Values[key]; !ok || got != want {
			t.Errorf("%s = %q (%v), want %q", key, got, ok, want)
		}
	}
	for _, key := range []string{"TF_GLM_EFFORT_TAIL", "TF_GLM_KV", "TF_GLM_STREAM_SMOOTH_MS"} {
		if !s.Exported[key] {
			t.Errorf("%s is exported upstream", key)
		}
	}
	if s.Exported["SPILL_GIB"] || s.Exported["HOST"] {
		t.Error("a launcher key was read as the server's environment")
	}
	// MASTER_ADDR is computed from the hostname, PARALLEL from DRAFTER and
	// TP, and TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS is set inside an if.
	for _, key := range []string{"MASTER_ADDR", "PARALLEL", "TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS"} {
		if v, ok := s.Values[key]; ok {
			t.Errorf("%s = %q, but upstream computes it", key, v)
		}
	}
}

func TestEnvShSkipsConditionalAndCommentedAssignments(t *testing.T) {
	s := parseEnvSh([]byte(strings.Join([]string{
		`A="${A:-1}"; [[ -n "$X" ]] && B="2"`,
		`C="${C:-3}"   # D="4"`,
		`  E="5"`,
		`F="$(hostname)"`,
		`F="${F:-6}"`,
		`G="${G:-$F}"`,
	}, "\n")))
	want := map[string]string{"A": "1", "C": "3", "F": "6", "G": "6"}
	if len(s.Values) != len(want) {
		t.Errorf("values = %v, want %v", s.Values, want)
	}
	for k, v := range want {
		if s.Values[k] != v {
			t.Errorf("%s = %q, want %q", k, s.Values[k], v)
		}
	}
}

// shippedEntry is the shipped recipe that cites the fixture, as JSON, so a
// test can change one claim and see the report move.
func shippedEntry(t *testing.T) map[string]any {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("..", "..", "catalogue", "recipes", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var r map[string]any
		if err := json.Unmarshal(b, &r); err != nil {
			t.Fatal(err)
		}
		if src, _ := r["source"].(map[string]any); src["format"] == "env-sh" {
			return r
		}
	}
	t.Fatal("no shipped recipe cites an env-sh source")
	return nil
}

func runOn(t *testing.T, entries ...map[string]any) (int, string) {
	t.Helper()
	dir := t.TempDir()
	for _, e := range entries {
		b, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, e["id"].(string)+".json"), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var out strings.Builder
	n, err := run(dir, envShFixtureDir, &out)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	return n, out.String()
}

func deviations(e map[string]any) []any {
	return e["source"].(map[string]any)["deviations"].([]any)
}

// The shipped entry's every departure from the file it cites is declared,
// and every declaration still describes the file.
func TestTheShippedEnvShEntryIsFullyDeclared(t *testing.T) {
	n, out := runOn(t, shippedEntry(t))
	if n != 0 {
		t.Fatalf("count = %d, want 0:\n%s", n, out)
	}
	for _, want := range []string{"config HOST", "0.0.0.0 -> 127.0.0.1", "env    TF_GLM_EFFORT_TAIL", "ours alone"} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q:\n%s", want, out)
		}
	}
}

// A config declaration is checked against upstream's default, so one that
// names a value upstream no longer has is reported, not trusted.
func TestAStaleConfigDeclarationIsReported(t *testing.T) {
	e := shippedEntry(t)
	for _, d := range deviations(e) {
		if d := d.(map[string]any); d["setting"] == "SPILL_GIB" {
			d["upstream"] = "64"
		}
	}
	n, out := runOn(t, e)
	if n != 1 || !strings.Contains(out, "MISDECLARED") {
		t.Errorf("count = %d, want the stale SPILL_GIB declaration:\n%s", n, out)
	}
}

func TestAnUndeclaredEnvChangeIsCounted(t *testing.T) {
	e := shippedEntry(t)
	env := e["env"].(map[string]any)
	env["TF_GLM_COPY_MAX"] = "30"
	delete(env, "TF_GLM_KV")
	n, out := runOn(t, e)
	if n != 2 {
		t.Errorf("count = %d, want 2:\n%s", n, out)
	}
	for _, want := range []string{"15 -> 30", "fp8 -> not used"} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q:\n%s", want, out)
		}
	}
}

// An entry whose pinned models the file does not pin cites the wrong file,
// or upstream moved its pins.
func TestAModelTheSourceDoesNotPinIsAWrongSource(t *testing.T) {
	e := shippedEntry(t)
	models := e["models"].([]any)
	models[0].(map[string]any)["revision"] = strings.Repeat("0", 40)
	n, out := runOn(t, e)
	if n != 1 || !strings.Contains(out, "WRONG SOURCE") {
		t.Errorf("count = %d:\n%s", n, out)
	}
}

func TestAManualSourceIsNotChecked(t *testing.T) {
	e := shippedEntry(t)
	e["source"].(map[string]any)["format"] = "manual"
	n, out := runOn(t, e)
	if n != 1 || !strings.Contains(out, `NOT CHECKED -- no reader for source format "manual"`) {
		t.Errorf("count = %d:\n%s", n, out)
	}
}

func TestDiffEnvShReportsWhatMoved(t *testing.T) {
	was := parseEnvSh([]byte("A=\"${A:-1}\"\nexport B=\"${B:-x}\"\nC=\"${C:-gone}\"\n"))
	now := parseEnvSh([]byte("A=\"${A:-2}\"\nexport B=\"${B:-x}\"\nexport D=\"${D:-new}\"\n"))
	got := diffEnvSh(was, now)
	want := []change{
		{"config", "A", "1", "2"},
		{"config", "C", "gone", "(removed)"},
		{"env", "D", "(absent)", "new"},
	}
	if len(got) != len(want) {
		t.Fatalf("changes = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("change %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}
