package config_test

import (
	"reflect"
	"testing"

	"github.com/coder/llm-helper/internal/config"
)

func TestSplitWords(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"   ", nil},
		{"--enable-auto-tool-choice", []string{"--enable-auto-tool-choice"}},
		{"--a --b  --c", []string{"--a", "--b", "--c"}},
		// JSON-valued flags with spaces must survive as one argument.
		{`--override-generation-config '{"temperature": 0.7}'`,
			[]string{"--override-generation-config", `{"temperature": 0.7}`}},
		{`--default-chat-template-kwargs "{\"thinking\": true}"`,
			[]string{"--default-chat-template-kwargs", `{"thinking": true}`}},
		// Quotes glued to unquoted text form a single word.
		{`--x=a" b"c`, []string{`--x=a b` + "c"}},
		{`a\ b`, []string{"a b"}},
		{"--tool-call-parser qwen3_coder", []string{"--tool-call-parser", "qwen3_coder"}},
		{"\ttabs\nand newlines ", []string{"tabs", "and", "newlines"}},
	}
	for _, c := range cases {
		got, err := config.SplitWords(c.in)
		if err != nil {
			t.Errorf("config.SplitWords(%q) error: %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("config.SplitWords(%q) = %#v, want %#v", c.in, got, c.want)
		}
	}
}

func TestSplitWordsErrors(t *testing.T) {
	for _, in := range []string{`"unterminated`, `'unterminated`, `trailing\`} {
		if _, err := config.SplitWords(in); err == nil {
			t.Errorf("config.SplitWords(%q) should have failed", in)
		}
	}
}

// Round-trip: anything Quote produces, SplitWords must read back as the
// single original value.
func TestQuoteSplitRoundTrip(t *testing.T) {
	values := []string{
		"plain",
		"two words",
		`{"temperature": 0.7}`,
		`nested "quotes" inside`,
		`back\slash`,
	}
	for _, v := range values {
		got, err := config.SplitWords(config.Quote(v))
		if err != nil {
			t.Errorf("round trip %q: %v", v, err)
			continue
		}
		if len(got) != 1 || got[0] != v {
			t.Errorf("round trip %q = %#v", v, got)
		}
	}
}

func TestSplitArgs(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{{
		// Profiles created before values were quoted -- and everything
		// 'adopt' reconstructs from a running server's argv -- store bare
		// JSON. Quote removal must not corrupt it.
		name: "legacy bare JSON survives",
		in:   `--default-chat-template-kwargs {"thinking":true}`,
		want: []string{"--default-chat-template-kwargs", `{"thinking":true}`},
	}, {
		name: "quoted JSON with spaces stays one argument",
		in:   `--override-generation-config '{"temperature": 0.7}'`,
		want: []string{"--override-generation-config", `{"temperature": 0.7}`},
	}, {
		name: "ordinary flags are untouched",
		in:   "--enable-auto-tool-choice --tool-call-parser poolside_v1",
		want: []string{"--enable-auto-tool-choice", "--tool-call-parser", "poolside_v1"},
	}, {
		// Not JSON, so POSIX quote removal still applies.
		name: "quoted plain text is still unquoted",
		in:   `--served-model-name "my model"`,
		want: []string{"--served-model-name", "my model"},
	}}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := config.SplitArgs(c.in)
			if err != nil {
				t.Fatalf("config.SplitArgs(%q) error: %v", c.in, err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("config.SplitArgs(%q) = %#v, want %#v", c.in, got, c.want)
			}
		})
	}
}

// The bare-JSON rule must not resurrect malformed input as valid.
func TestSplitArgsErrors(t *testing.T) {
	if _, err := config.SplitArgs(`--x '{"unterminated": 1}`); err == nil {
		t.Error("unterminated quote should still fail")
	}
}
