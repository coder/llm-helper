//go:build darwin

package detect

import (
	"bytes"
	"encoding/binary"
	"os"
	"os/exec"
	"testing"
	"time"
)

// buildProcArgs assembles a KERN_PROCARGS2 buffer the way the kernel does.
func buildProcArgs(execPath string, args []string, env []string) []byte {
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, uint32(len(args)))
	b.WriteString(execPath)
	b.WriteByte(0)
	b.WriteByte(0) // alignment padding the kernel inserts
	for _, a := range args {
		b.WriteString(a)
		b.WriteByte(0)
	}
	for _, e := range env {
		b.WriteString(e)
		b.WriteByte(0)
	}
	return b.Bytes()
}

func TestParseProcArgs(t *testing.T) {
	want := []string{
		"/opt/venv/bin/vllm", "serve", "Qwen/Qwen3-0.6B",
		"--chat-template", "/Volumes/My Drive/tpl.jinja",
		"--default-chat-template-kwargs", `{"thinking": true}`,
	}
	buf := buildProcArgs("/opt/venv/bin/vllm", want, []string{"PATH=/usr/bin", "HOME=/Users/x"})

	got, err := parseProcArgs(buf)
	if err != nil {
		t.Fatalf("parseProcArgs: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d args %q, want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("arg %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestParseProcArgsStopsBeforeEnvironment(t *testing.T) {
	buf := buildProcArgs("/bin/x", []string{"/bin/x", "serve"}, []string{"SECRET=shh"})
	got, err := parseProcArgs(buf)
	if err != nil {
		t.Fatalf("parseProcArgs: %v", err)
	}
	for _, a := range got {
		if a == "SECRET=shh" {
			t.Fatalf("environment leaked into argv: %q", got)
		}
	}
}

func TestParseProcArgsRejectsGarbage(t *testing.T) {
	for name, buf := range map[string][]byte{
		"empty":     {},
		"short":     {1, 2},
		"zero argc": buildProcArgs("/bin/x", nil, nil),
	} {
		if _, err := parseProcArgs(buf); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

// TestProcArgsLive checks the syscall against a real process, which is the
// part a hand-built buffer can't cover.
func TestProcArgsLive(t *testing.T) {
	want := []string{"/bin/sleep", "5"}
	cmd := exec.Command("/bin/sleep", "5")
	if err := cmd.Start(); err != nil {
		t.Skipf("couldn't start a probe process: %v", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}()
	time.Sleep(200 * time.Millisecond)

	got, err := procArgs(cmd.Process.Pid)
	if err != nil {
		t.Fatalf("procArgs(%d): %v", cmd.Process.Pid, err)
	}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// TestProcArgsForeignProcess documents that the syscall is not permitted for
// other users' processes, which is why ps stays as the fallback.
func TestProcArgsForeignProcess(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root, which can read any process")
	}
	// pid 1 (launchd) is root-owned.
	if _, err := procArgs(1); err == nil {
		t.Log("procArgs(1) unexpectedly succeeded; the ps fallback is still correct")
	}
}

func TestLabelInServices(t *testing.T) {
	t.Parallel()

	const text = "system = {\n\ttype = system\n\tservices = {\n\t\t     338      - \tcom.apple.driver.Example\n\t\t    7151      0 \tcom.coder.vllm\n\t}\n\tdisabled services = {\n\t\t\"com.example.off\" => disabled\n\t}\n}\n"
	if got := labelInServices(text, 7151); got != "com.coder.vllm" {
		t.Errorf("labelInServices(7151) = %q, want com.coder.vllm", got)
	}
	if got := labelInServices(text, 42); got != "" {
		t.Errorf("labelInServices(42) = %q, want none", got)
	}
}
