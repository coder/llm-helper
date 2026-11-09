// Package modeltype reads the model-type definitions in catalogue/model-types/:
// what every repo with a given HuggingFace model_type needs to run correctly
// on each engine. A machine owner can add or replace one without a rebuild by
// dropping a file into model-types.d in the config directory; see UseLocal.
package modeltype

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"strings"

	"github.com/coder/llm-helper/catalogue"
	"github.com/coder/llm-helper/internal/engine"
)

// SchemaVersion is the version of the file format this build reads.
const SchemaVersion = 1

// Entry is one model type.
type Entry struct {
	// ID is the model_type value in a repo's config.json, and the file name.
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	Why         string `json:"why"`
	// Engines holds what this model type needs on each engine, by engine id.
	Engines map[string]Rules `json:"engines,omitempty"`
	// Thinking, when set, describes the model's chat template switch for
	// reasoning before it answers.
	Thinking *Thinking `json:"thinking,omitempty"`
	// Source is the model-types.d file a local entry came from, and "" for
	// a shipped one.
	Source string `json:"-"`
}

// Rules is what a model type needs when one engine serves it.
type Rules struct {
	Files             []FileSet     `json:"files,omitempty"`
	Args              []Arg         `json:"args,omitempty"`
	ChatTemplateFixes []TemplateFix `json:"chat_template_fixes,omitempty"`
}

// FileSet is files fetched into a directory under the data directory before
// the model is served.
type FileSet struct {
	Dir string `json:"dir"`
	// Env, when set, is put in the profile pointing at the directory.
	Env     string   `json:"env,omitempty"`
	BaseURL string   `json:"base_url"`
	Names   []string `json:"names"`
	Why     string   `json:"why"`
}

// Thinking is a chat template kwarg that turns reasoning on or off.
type Thinking struct {
	Kwarg string `json:"kwarg"`
	// Default is whether the model thinks when the kwarg is not passed.
	Default bool `json:"default"`
	// Detect is DetectTemplate or DetectAlways.
	Detect string `json:"detect,omitempty"`
	Why    string `json:"why"`
}

const (
	// DetectTemplate applies the switch only to repos whose chat template
	// names the kwarg; it is the default.
	DetectTemplate = "template"
	// DetectAlways applies it to every repo of the model type, for types
	// whose chat formatting lives in the engine rather than a template.
	DetectAlways = "always"
)

// Applies reports whether the switch is in template, a repo's chat template
// ("" when it has none).
func (t Thinking) Applies(template string) bool {
	if t.Detect == DetectAlways {
		return true
	}
	return regexp.MustCompile(`\b` + regexp.QuoteMeta(t.Kwarg) + `\b`).MatchString(template)
}

// Arg is a flag added to a new profile unless it already names it. Values
// are tried in order and the first the installed engine accepts is used; with
// none the flag is added bare. With is added alongside it.
type Arg struct {
	Flag   string   `json:"flag"`
	Values []string `json:"values,omitempty"`
	With   []string `json:"with,omitempty"`
	Why    string   `json:"why"`
}

// TemplateFix patches a model's bundled chat template when it contains
// Contains, replacing Match (a regular expression) with Replace.
type TemplateFix struct {
	Contains string `json:"contains"`
	Match    string `json:"match"`
	Replace  string `json:"replace"`
	Why      string `json:"why"`
}

// Apply returns the patched template, or ok false when the template does not
// contain the text this fix is for. It is an error for the text to be there
// and the pattern not to match, so a fix is never claimed that did not apply.
func (f TemplateFix) Apply(template string) (patched string, ok bool, err error) {
	if !strings.Contains(template, f.Contains) {
		return "", false, nil
	}
	re, err := regexp.Compile(f.Match)
	if err != nil {
		return "", false, err
	}
	out := re.ReplaceAllString(template, f.Replace)
	if out == template {
		return "", false, fmt.Errorf("the template contains %q but not in the shape the fix matches; patch it by hand", f.Contains)
	}
	return out, true, nil
}

// Registry is a set of validated entries.
type Registry struct {
	entries map[string]Entry
}

// Lookup returns the entry for a model_type.
func (r *Registry) Lookup(modelType string) (Entry, bool) {
	e, ok := r.entries[modelType]
	return e, ok
}

// RulesFor returns what modelType needs on engineID, empty when nothing.
func (r *Registry) RulesFor(modelType, engineID string) Rules {
	e, ok := r.Lookup(modelType)
	if !ok {
		return Rules{}
	}
	return e.Engines[engineID]
}

// List returns every entry, sorted by id.
func (r *Registry) List() []Entry {
	out := make([]Entry, 0, len(r.entries))
	for _, e := range r.entries {
		out = append(out, e)
	}
	slices.SortFunc(out, func(a, b Entry) int { return strings.Compare(a.ID, b.ID) })
	return out
}

// with returns a copy of r with e added, replacing any entry with its id.
func (r *Registry) with(e Entry) *Registry {
	next := &Registry{entries: make(map[string]Entry, len(r.entries)+1)}
	for id, old := range r.entries {
		next.entries[id] = old
	}
	next.entries[e.ID] = e
	return next
}

// Load reads and validates every *.json file in fsys, each named after the
// id it defines.
func Load(fsys fs.FS) (*Registry, error) {
	names, err := fs.Glob(fsys, "*.json")
	if err != nil {
		return nil, err
	}
	reg := &Registry{entries: map[string]Entry{}}
	var errs []error
	for _, name := range names {
		data, err := fs.ReadFile(fsys, name)
		var e Entry
		if err == nil {
			e, err = parse(name, data)
		}
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, err))
			continue
		}
		reg = reg.with(e)
	}
	return reg, errors.Join(errs...)
}

var (
	idPattern    = regexp.MustCompile(`^[a-z0-9][a-z0-9_.-]*$`)
	envPattern   = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	flagPattern  = regexp.MustCompile(`^--?[A-Za-z0-9][A-Za-z0-9-]*$`)
	namePattern  = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)
	kwargPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

// parse decodes one file strictly and checks it, so a typo fails when the
// file loads rather than when a model is pulled.
func parse(name string, data []byte) (Entry, error) {
	var f struct {
		SchemaVersion int `json:"schema_version"`
		Entry
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return Entry{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return Entry{}, errors.New("unexpected data after the top-level object")
	}
	if f.SchemaVersion != SchemaVersion {
		return Entry{}, fmt.Errorf("schema_version is %d; this llm-helper reads version %d", f.SchemaVersion, SchemaVersion)
	}
	e := f.Entry
	if base := strings.TrimSuffix(path.Base(name), ".json"); e.ID != base {
		return Entry{}, fmt.Errorf("the file is named %s.json but defines %q; name it after the model_type it describes", base, e.ID)
	}
	if !idPattern.MatchString(e.ID) {
		return Entry{}, fmt.Errorf("id %q is not a model_type", e.ID)
	}
	if e.Why == "" {
		return Entry{}, errors.New("why is empty")
	}
	if len(e.Engines) == 0 && e.Thinking == nil {
		return Entry{}, errors.New("engines and thinking are both empty")
	}
	if t := e.Thinking; t != nil {
		switch {
		case !kwargPattern.MatchString(t.Kwarg):
			return Entry{}, fmt.Errorf("thinking.kwarg %q is not a template variable name", t.Kwarg)
		case t.Detect != "" && t.Detect != DetectTemplate && t.Detect != DetectAlways:
			return Entry{}, fmt.Errorf("thinking.detect %q must be %s or %s", t.Detect, DetectTemplate, DetectAlways)
		case t.Why == "":
			return Entry{}, errors.New("thinking.why is empty")
		}
	}
	for id, rules := range e.Engines {
		prov, err := engine.Lookup(id)
		if err != nil {
			return Entry{}, fmt.Errorf("engines: %w", err)
		}
		if !prov.Capabilities().VLLMArgs && (len(rules.Args) > 0 || len(rules.ChatTemplateFixes) > 0) {
			return Entry{}, fmt.Errorf("engines.%s: args and chat_template_fixes apply only to engines with the vllm_args capability; %s takes files", id, id)
		}
		if err := rules.check(); err != nil {
			return Entry{}, fmt.Errorf("engines.%s: %w", id, err)
		}
	}
	return e, nil
}

func (r Rules) check() error {
	for i, f := range r.Files {
		switch {
		case f.Dir == "" || path.IsAbs(f.Dir) || path.Clean(f.Dir) != f.Dir || strings.HasPrefix(f.Dir, ".."):
			return fmt.Errorf("files[%d].dir %q must be a plain path under the data directory", i, f.Dir)
		case f.Env != "" && !envPattern.MatchString(f.Env):
			return fmt.Errorf("files[%d].env %q is not an environment variable name", i, f.Env)
		case !strings.HasPrefix(f.BaseURL, "https://") || !strings.HasSuffix(f.BaseURL, "/"):
			return fmt.Errorf("files[%d].base_url must be an https URL ending in /", i)
		case len(f.Names) == 0:
			return fmt.Errorf("files[%d].names is empty", i)
		case f.Why == "":
			return fmt.Errorf("files[%d].why is empty", i)
		}
		for _, n := range f.Names {
			if !namePattern.MatchString(n) || n == "." || n == ".." {
				return fmt.Errorf("files[%d].names: %q is not a file name", i, n)
			}
		}
	}
	for i, a := range r.Args {
		if !flagPattern.MatchString(a.Flag) {
			return fmt.Errorf("args[%d].flag %q is not a flag", i, a.Flag)
		}
		if a.Why == "" {
			return fmt.Errorf("args[%d].why is empty", i)
		}
		for _, w := range a.With {
			if !flagPattern.MatchString(w) {
				return fmt.Errorf("args[%d].with: %q is not a flag", i, w)
			}
		}
	}
	for i, t := range r.ChatTemplateFixes {
		if t.Contains == "" || t.Why == "" {
			return fmt.Errorf("chat_template_fixes[%d] needs contains and why", i)
		}
		if _, err := regexp.Compile(t.Match); err != nil {
			return fmt.Errorf("chat_template_fixes[%d].match: %w", i, err)
		}
	}
	return nil
}

var shippedRegistry = mustLoad(catalogue.ModelTypes)

func mustLoad(fsys fs.FS) *Registry {
	reg, err := Load(fsys)
	if err != nil {
		panic("catalogue/model-types: " + err.Error())
	}
	return reg
}
