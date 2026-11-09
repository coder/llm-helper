package modeltype

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/coder/llm-helper/catalogue"
)

func TestTheShippedModelTypesLoad(t *testing.T) {
	reg, err := Load(catalogue.ModelTypes)
	if err != nil {
		t.Fatal(err)
	}
	if len(reg.List()) == 0 {
		t.Fatal("no model types shipped")
	}
}

func readTemplate(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "hf", "testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestTheShippedQwen35FixPatchesItsTemplate(t *testing.T) {
	for _, id := range []string{"qwen3_5", "qwen3_5_moe"} {
		fixes := Default().RulesFor(id, "vllm").ChatTemplateFixes
		if len(fixes) != 1 {
			t.Fatalf("%s: %d fixes, want 1", id, len(fixes))
		}
		patched, ok, err := fixes[0].Apply(readTemplate(t, "chat_template_broken.jinja"))
		if err != nil || !ok {
			t.Fatalf("%s: ok %v, err %v", id, ok, err)
		}
		if strings.Contains(patched, "raise_exception") {
			t.Errorf("%s: the patched template still raises", id)
		}
		if !strings.Contains(patched, `message.role == "system"`) {
			t.Errorf("%s: the rest of the template should be left alone", id)
		}
		if _, ok, err := fixes[0].Apply(readTemplate(t, "chat_template_clean.jinja")); ok || err != nil {
			t.Errorf("%s: a clean template got ok %v, err %v", id, ok, err)
		}
	}
}

// The text being there and the pattern not matching means the fix would be
// claimed without applying.
func TestAFixThatDoesNotMatchItsShapeIsAnError(t *testing.T) {
	f := Default().RulesFor("qwen3_5", "vllm").ChatTemplateFixes[0]
	if _, ok, err := f.Apply(f.Contains + " (mentioned in a comment)"); ok || err == nil {
		t.Errorf("ok %v, err %v, want an error", ok, err)
	}
}

const good = `{"schema_version":1,"id":"toy","why":"w","engines":{"vllm":{"args":[{"flag":"--x","why":"w"}]}}}`

func TestBadFilesAreRefused(t *testing.T) {
	for name, c := range map[string]struct{ body, want string }{
		"misnamed":       {good, "defines \"toy\""},
		"bad id":         {`{"schema_version":1,"id":"bad id","why":"w","engines":{"vllm":{}}}`, "is not a model_type"},
		"unknown_field":  {`{"schema_version":1,"id":"unknown_field","why":"w","engines":{},"extra":1}`, "unknown field"},
		"unknown_engine": {`{"schema_version":1,"id":"unknown_engine","why":"w","engines":{"nope":{}}}`, "unknown PROVIDER"},
		"mlx_args":       {`{"schema_version":1,"id":"mlx_args","why":"w","engines":{"mlx":{"args":[{"flag":"--x","why":"w"}]}}}`, "vllm_args capability"},
		"bad_regex":      {`{"schema_version":1,"id":"bad_regex","why":"w","engines":{"vllm":{"chat_template_fixes":[{"contains":"c","match":"(","replace":"","why":"w"}]}}}`, "match"},
		"escaping_dir":   {`{"schema_version":1,"id":"escaping_dir","why":"w","engines":{"vllm":{"files":[{"dir":"../x","base_url":"https://a/","names":["f"],"why":"w"}]}}}`, "plain path"},
		"http_url":       {`{"schema_version":1,"id":"http_url","why":"w","engines":{"vllm":{"files":[{"dir":"x","base_url":"http://a/","names":["f"],"why":"w"}]}}}`, "https URL"},
		"no_why":         {`{"schema_version":1,"id":"no_why","engines":{"vllm":{}}}`, "why is empty"},
		"wrong_version":  {`{"schema_version":2,"id":"wrong_version","why":"w","engines":{"vllm":{}}}`, "schema_version"},
		"empty":          {`{"schema_version":1,"id":"empty","why":"w"}`, "engines and thinking are both empty"},
		"bad_kwarg":      {`{"schema_version":1,"id":"bad_kwarg","why":"w","thinking":{"kwarg":"enable thinking","why":"w"}}`, "thinking.kwarg"},
		"bad_detect":     {`{"schema_version":1,"id":"bad_detect","why":"w","thinking":{"kwarg":"t","detect":"sometimes","why":"w"}}`, "thinking.detect"},
		"thinking_why":   {`{"schema_version":1,"id":"thinking_why","why":"w","thinking":{"kwarg":"t"}}`, "thinking.why is empty"},
	} {
		fsys := fstest.MapFS{name + ".json": {Data: []byte(c.body)}}
		if _, err := Load(fsys); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want one mentioning %q", name, err, c.want)
		}
	}
}

func TestAThinkingOnlyEntryLoads(t *testing.T) {
	body := `{"schema_version":1,"id":"toy","why":"w","thinking":{"kwarg":"enable_thinking","default":true,"why":"w"}}`
	reg, err := Load(fstest.MapFS{"toy.json": {Data: []byte(body)}})
	if err != nil {
		t.Fatal(err)
	}
	e, ok := reg.Lookup("toy")
	if !ok || e.Thinking == nil || !e.Thinking.Default || len(e.Engines) != 0 {
		t.Errorf("toy = %+v, %v, want a thinking switch and no engines", e, ok)
	}
}

func TestThinkingApplies(t *testing.T) {
	for _, c := range []struct {
		name     string
		t        Thinking
		template string
		want     bool
	}{
		{"kwarg in the template", Thinking{Kwarg: "enable_thinking"}, "{%- if enable_thinking is defined %}", true},
		{"kwarg absent", Thinking{Kwarg: "enable_thinking"}, "{{ messages }}", false},
		{"only part of a longer name", Thinking{Kwarg: "thinking"}, "{% if enable_thinking %}", false},
		{"no template", Thinking{Kwarg: "thinking"}, "", false},
		{"always, with no template", Thinking{Kwarg: "thinking", Detect: DetectAlways}, "", true},
	} {
		if got := c.t.Applies(c.template); got != c.want {
			t.Errorf("%s: Applies = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestTheShippedThinkingSwitches(t *testing.T) {
	for id, want := range map[string]Thinking{
		"qwen3":         {Kwarg: "enable_thinking", Default: true},
		"qwen3_moe":     {Kwarg: "enable_thinking", Default: true},
		"qwen3_5":       {Kwarg: "enable_thinking", Default: true},
		"qwen3_5_moe":   {Kwarg: "enable_thinking", Default: true},
		"glm4_moe":      {Kwarg: "enable_thinking", Default: true},
		"glm4_moe_lite": {Kwarg: "enable_thinking", Default: true},
		"glm_moe_dsa":   {Kwarg: "enable_thinking", Default: true},
		"glm4v_moe":     {Kwarg: "enable_thinking", Default: true},
		"smollm3":       {Kwarg: "enable_thinking", Default: true},
		"deepseek_v3":   {Kwarg: "thinking", Default: false},
		"deepseek_v4":   {Kwarg: "thinking", Default: true, Detect: DetectAlways},
	} {
		e, ok := Default().Lookup(id)
		if !ok || e.Thinking == nil {
			t.Errorf("%s: no thinking switch shipped", id)
			continue
		}
		got := *e.Thinking
		got.Why = ""
		if got.Detect == DetectTemplate {
			got.Detect = ""
		}
		if got != want {
			t.Errorf("%s: thinking = %+v, want %+v", id, got, want)
		}
	}
}

func TestALocalFileAddsOrReplaces(t *testing.T) {
	dir := t.TempDir()
	replaced := `{"schema_version":1,"id":"gpt_oss","why":"local","engines":{"vllm":{}}}`
	for name, body := range map[string]string{
		"toy.json":     good,
		"gpt_oss.json": replaced,
		"broken.json":  "{",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	errs := UseLocal(dir)
	t.Cleanup(func() { UseLocal(t.TempDir()) })
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "broken.json") {
		t.Errorf("errs = %v, want only broken.json reported", errs)
	}
	if e, ok := Default().Lookup("toy"); !ok || e.Source != filepath.Join(dir, "toy.json") {
		t.Errorf("toy = %+v, %v, want it added with its source", e, ok)
	}
	if e, _ := Default().Lookup("gpt_oss"); e.Why != "local" {
		t.Errorf("gpt_oss why = %q, want the local file to replace the shipped one", e.Why)
	}
	if _, ok := Default().Lookup("qwen3_5"); !ok {
		t.Error("a local file removed an unrelated shipped entry")
	}
}

// What a model type needs belongs in its catalogue file, so Go code must not
// name one or repeat what its file says.
func TestNoGoCodeRepeatsAShippedModelType(t *testing.T) {
	reg, err := Load(catalogue.ModelTypes)
	if err != nil {
		t.Fatal(err)
	}
	var needles []string
	for _, e := range reg.List() {
		needles = append(needles, `"`+e.ID+`"`)
		for _, r := range e.Engines {
			for _, f := range r.Files {
				needles = append(needles, f.Names...)
				needles = append(needles, f.BaseURL)
				if f.Env != "" {
					needles = append(needles, f.Env)
				}
			}
			for _, a := range r.Args {
				for _, v := range a.Values {
					needles = append(needles, `"`+v+`"`)
				}
			}
			for _, x := range r.ChatTemplateFixes {
				needles = append(needles, x.Contains)
			}
		}
	}
	root := filepath.Join("..", "..", "internal")
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, n := range needles {
			if strings.Contains(string(b), n) {
				t.Errorf("%s contains %s, which belongs in catalogue/model-types", path, n)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
