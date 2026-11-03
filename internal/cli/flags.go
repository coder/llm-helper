package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"golang.org/x/term"
)

// stdoutIsTTY reports whether stdout is a terminal. When it isn't (pipes,
// cron, a UI backend), commands default to non-interactive behavior.
func stdoutIsTTY() bool { return isTerminal(os.Stdout) }

// stdinIsTTY reports whether stdin can actually answer a prompt.
func stdinIsTTY() bool { return isTerminal(os.Stdin) }

// isTerminal asks the kernel instead of reading the file mode. /dev/null is a
// character device just as a terminal is, so a mode check calls it one and
// sends 'logs > /dev/null' into a tail that never returns.
func isTerminal(f *os.File) bool {
	return term.IsTerminal(int(f.Fd()))
}

// printJSON renders v as indented JSON on stdout.
func printJSON(v any) error { return printJSONTo(os.Stdout, v) }

// printJSONTo renders v as indented JSON on w.
func printJSONTo(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return fmt.Errorf("encoding JSON: %w", err)
	}
	return nil
}
