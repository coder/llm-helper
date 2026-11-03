package detect

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/coder/vllm-helper/internal/hardware"
)

// fixtureDir holds the hardware reports the definitions are tested against.
// It sits beside definitions.json so a contribution touches one directory.
const fixtureDir = "../hardware/testdata"

type fixture struct {
	name string
	Report
}

func loadFixtures(t *testing.T) []fixture {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(fixtureDir, "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatalf("no fixtures in %s", fixtureDir)
	}
	var out []fixture
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.DisallowUnknownFields()
		var r Report
		if err := dec.Decode(&r); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		out = append(out, fixture{name: filepath.Base(p), Report: r})
	}
	return out
}

// Each fixture is a machine's readings and what llm-helper should conclude
// from them. A change to detection or to a definition that changes any
// conclusion fails here, naming the machine it changed it for.
func TestFixturesDetectWhatTheyRecord(t *testing.T) {
	for _, f := range loadFixtures(t) {
		t.Run(f.name, func(t *testing.T) {
			if f.Readings.SchemaVersion != hardware.SchemaVersion {
				t.Errorf("schema_version %d, want %d", f.Readings.SchemaVersion, hardware.SchemaVersion)
			}
			// Measured and synthetic readings must be told apart, so nobody
			// cites an invented machine as evidence.
			if !strings.HasPrefix(f.Provenance, "measured: ") && !strings.HasPrefix(f.Provenance, "synthetic: ") {
				t.Errorf("provenance %q must start with \"measured: \" or \"synthetic: \"", f.Provenance)
			}
			got := DetectedFrom(FromReadings(f.Readings))
			if !reflect.DeepEqual(got, f.Detected) {
				want, _ := json.MarshalIndent(f.Detected, "", "  ")
				have, _ := json.MarshalIndent(got, "", "  ")
				t.Errorf("detected:\n%s\nwant:\n%s", have, want)
			}
		})
	}
}

// Every shipped definition must be reached by at least one fixture. That also
// proves no earlier definition shadows it: the fixture's GPU would match the
// earlier one first, and detect its id instead.
func TestEveryDefinitionHasAFixture(t *testing.T) {
	seen := map[string]bool{}
	for _, f := range loadFixtures(t) {
		seen[f.Detected.HardwareID] = true
	}
	for _, d := range hardware.Shipped().Definitions() {
		if !seen[d.ID] {
			t.Errorf("no fixture in %s detects %s; add one, measured with 'llm-helper hardware report' if you can", fixtureDir, d.ID)
		}
	}
}
