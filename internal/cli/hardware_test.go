package cli_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/hardware"
)

// A hardware.d file is loaded before every command, so 'hardware list' shows
// what a machine owner added and where it came from, and a broken file is
// named rather than taking the others with it.
func TestRunHardwareListShowsLocalDefinitions(t *testing.T) {
	configDir, _ := newTree(t)
	dir := filepath.Join(configDir, "hardware.d")
	good := filepath.Join(dir, "machines", "acme_box.json")
	files := map[string]string{
		filepath.Join(dir, "machines", "broken.json"): `{"schema_version":1,"id":`,
		filepath.Join(dir, "vendors", "acme.json"):    `{"schema_version":1,"id":"acme","name":"Acme"}`,
		good: `{"schema_version":1,"id":"acme_box","display_name":"Acme box","match":{"vendor":"acme","gpu_name":"Acme"},"memory":"unified","why":"w"}`,
	}
	for path, body := range files {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Later tests in this process must not see this directory's definitions.
	t.Cleanup(func() { hardware.UseLocal(filepath.Join(dir, "gone")) })

	code, stdout, stderr := runCLI(t, "hardware", "list")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, stderr)
	}
	if !strings.Contains(stdout, "dgx_spark_gb10") || !strings.Contains(stdout, "shipped") {
		t.Errorf("the shipped definitions are missing:\n%s", stdout)
	}
	var acme string
	for _, line := range strings.Split(stdout, "\n") {
		if strings.HasPrefix(line, "acme_box") {
			acme = line
		}
	}
	if !strings.HasSuffix(acme, good) {
		t.Errorf("acme_box row = %q, want it to end with its file %s", acme, good)
	}
	if !strings.Contains(stderr, "broken.json") {
		t.Errorf("stderr = %q, want the broken file named", stderr)
	}

	code, stdout, _ = runCLI(t, "hardware", "list", "--json")
	if code != 0 {
		t.Fatalf("--json exit %d", code)
	}
	var rows []struct {
		ID     string `json:"id"`
		Source string `json:"source"`
	}
	if err := json.Unmarshal([]byte(stdout), &rows); err != nil {
		t.Fatalf("--json is not JSON: %v\n%s", err, stdout)
	}
	if len(rows) == 0 || rows[len(rows)-1].ID != "acme_box" || rows[len(rows)-1].Source != good {
		t.Errorf("--json rows = %+v, want acme_box last, from %s", rows, good)
	}
}

// A report is meant to be pasted into a pull request, so it must parse and
// must carry nothing that names the host it came from.
func TestRunHardwareReportNamesNoHost(t *testing.T) {
	newTree(t)
	code, stdout, stderr := runCLI(t, "hardware", "report")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, stderr)
	}
	var r struct {
		Provenance string         `json:"provenance"`
		Readings   map[string]any `json:"readings"`
	}
	if err := json.Unmarshal([]byte(stdout), &r); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, stdout)
	}
	if r.Provenance == "" || r.Readings["schema_version"] == nil {
		t.Errorf("report lacks provenance or schema_version:\n%s", stdout)
	}
	if host, err := os.Hostname(); err == nil && len(host) > 3 && strings.Contains(stdout, host) {
		t.Errorf("report contains the hostname %q:\n%s", host, stdout)
	}
	for _, key := range []string{"uuid", "serial", "hostname", "bus_id"} {
		if strings.Contains(strings.ToLower(stdout), key) {
			t.Errorf("report contains %q:\n%s", key, stdout)
		}
	}
}
