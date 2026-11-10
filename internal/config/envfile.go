package config

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// EnvFile is an ordered KEY=VALUE file, bash-sourceable, with comments
// preserved on read-modify-write. It is the shared on-disk format for
// config.env, secrets.env, model profiles, and the generated instance files --
// both this binary and the legacy bash scripts read the same files.
type EnvFile struct {
	// Lines preserves the original file verbatim (comments, blanks, order).
	Lines []string
	vals  map[string]string
}

// ParseEnvFile reads a bash-style env file. Values may be double-quoted
// (quotes are stripped); everything after the first '=' belongs to the value.
func ParseEnvFile(path string) (*EnvFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseEnv(string(data)), nil
}

func parseEnv(content string) *EnvFile {
	f := &EnvFile{vals: map[string]string{}}
	for line := range strings.SplitSeq(strings.TrimRight(content, "\n"), "\n") {
		f.Lines = append(f.Lines, line)
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		key, val, ok := strings.Cut(trimmed, "=")
		if !ok || key == "" || strings.ContainsAny(key, " \t") {
			continue
		}
		f.vals[key] = unquote(val)
	}
	return f
}

func unquote(v string) string {
	v = strings.TrimSpace(v)
	if len(v) < 2 {
		return v
	}
	if v[0] == '\'' && v[len(v)-1] == '\'' {
		return v[1 : len(v)-1]
	}
	if v[0] == '"' && v[len(v)-1] == '"' {
		// Bash double-quote semantics: \" and \\ are escapes.
		inner := v[1 : len(v)-1]
		var b strings.Builder
		b.Grow(len(inner))
		for i := 0; i < len(inner); i++ {
			if inner[i] == '\\' && i+1 < len(inner) && (inner[i+1] == '"' || inner[i+1] == '\\') {
				i++
			}
			b.WriteByte(inner[i])
		}
		return b.String()
	}
	return v
}

// Quote renders a value so the file stays bash-sourceable: multi-word
// values (like EXTRA_ARGS) must be double-quoted or bash treats the second
// word as a command when sourcing, and embedded quotes and backslashes
// must be escaped (bash treats an unquoted backslash as an escape).
func Quote(v string) string {
	// Reading trims whitespace, so anything the trim would eat must be quoted.
	if v == strings.TrimSpace(v) && !strings.ContainsAny(v, " \t\"\\'") {
		return v
	}
	escaped := strings.ReplaceAll(v, `\`, `\\`)
	escaped = strings.ReplaceAll(escaped, `"`, `\"`)
	return `"` + escaped + `"`
}

func (f *EnvFile) Get(key string) string { return f.vals[key] }

func (f *EnvFile) Has(key string) bool {
	_, ok := f.vals[key]
	return ok
}

// Set updates a key in place (preserving its position and surrounding
// comments) or appends it if absent.
func (f *EnvFile) Set(key, value string) {
	if f.vals == nil {
		f.vals = map[string]string{}
	}
	f.vals[key] = value
	rendered := key + "=" + Quote(value)
	for i, line := range f.Lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, key+"=") {
			f.Lines[i] = rendered
			return
		}
	}
	f.Lines = append(f.Lines, rendered)
}

// Delete removes a key's line entirely.
func (f *EnvFile) Delete(key string) {
	delete(f.vals, key)
	kept := f.Lines[:0]
	for _, line := range f.Lines {
		if strings.HasPrefix(strings.TrimSpace(line), key+"=") {
			continue
		}
		kept = append(kept, line)
	}
	f.Lines = kept
}

// Keys returns all defined keys, sorted for deterministic output.
func (f *EnvFile) Keys() []string {
	keys := make([]string, 0, len(f.vals))
	for k := range f.vals {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Write saves the file verbatim (comments and order preserved).
func (f *EnvFile) Write(path string, perm os.FileMode) error {
	content := strings.Join(f.Lines, "\n") + "\n"
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(content), perm); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LeadingComments returns the comment block at the top of the file (used
// for a profile's status/notes header).
func (f *EnvFile) LeadingComments() []string {
	var out []string
	for _, line := range f.Lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			out = append(out, trimmed)
			continue
		}
		break
	}
	return out
}

// MergedEnv concatenates env files into "KEY=VALUE" pairs suitable for a
// process environment; later files win on conflicts.
func MergedEnv(files ...*EnvFile) []string {
	merged := map[string]string{}
	for _, f := range files {
		if f == nil {
			continue
		}
		for k, v := range f.vals {
			merged[k] = v
		}
	}
	keys := make([]string, 0, len(merged))
	for k := range merged {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, fmt.Sprintf("%s=%s", k, merged[k]))
	}
	return out
}
