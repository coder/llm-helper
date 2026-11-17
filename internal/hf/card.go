package hf

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// cardFields are the model card header fields the rest of llm-helper reads.
type cardFields struct {
	baseModel   []string
	license     string
	pipeline    string
	tags        []string
	hasGGUFFile bool
}

// parseCard reads the YAML header of a model card: top-level scalars and
// lists, which is all those fields are.
func parseCard(text string) cardFields {
	var c cardFields
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return c
	}
	var key string
	for _, line := range lines[1:] {
		if strings.TrimSpace(line) == "---" {
			break
		}
		if item, ok := strings.CutPrefix(strings.TrimSpace(line), "- "); ok && key != "" {
			c.add(key, unquoteCard(item))
			continue
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok || line != strings.TrimLeft(line, " \t") {
			continue
		}
		key = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		switch {
		case v == "":
		case strings.HasPrefix(v, "[") && strings.HasSuffix(v, "]"):
			for item := range strings.SplitSeq(v[1:len(v)-1], ",") {
				c.add(key, unquoteCard(item))
			}
		default:
			c.add(key, unquoteCard(v))
		}
	}
	return c
}

func unquoteCard(v string) string {
	return strings.Trim(strings.TrimSpace(v), `"'`)
}

func (c *cardFields) add(key, value string) {
	if value == "" {
		return
	}
	switch key {
	case "base_model":
		c.baseModel = append(c.baseModel, value)
	case "license":
		c.license = value
	case "pipeline_tag":
		c.pipeline = value
	case "tags":
		c.tags = append(c.tags, value)
	}
}

// cachedModel is what the Hub would say about a cached repo, as far as its
// card and files tell.
func cachedModel(dir, repo string) Model {
	m := Model{ID: repo}
	raw, _ := os.ReadFile(filepath.Join(dir, "README.md"))
	c := parseCard(string(raw))
	m.PipelineTag = c.pipeline
	m.Tags = c.tags
	if c.license != "" {
		m.Tags = append(m.Tags, "license:"+c.license)
	}
	switch len(c.baseModel) {
	case 0:
	case 1:
		m.CardData.BaseModel, _ = json.Marshal(c.baseModel[0])
	default:
		m.CardData.BaseModel, _ = json.Marshal(c.baseModel)
	}
	_ = filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".gguf") {
			m.Tags = append(m.Tags, "gguf")
			return filepath.SkipAll
		}
		return nil
	})
	return m
}
