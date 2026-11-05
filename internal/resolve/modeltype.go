package resolve

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/hf"
	"github.com/coder/llm-helper/internal/modeltype"
	"github.com/coder/llm-helper/internal/vllm"
)

var httpClient = &http.Client{Timeout: 5 * time.Minute}

// modelTypeArgs returns the model type's flags this engine accepts that
// present does not already name, and a note for each. A flag or value the
// installed engine is not known to register is skipped, so an unreadable
// registry adds nothing.
func modelTypeArgs(args []modeltype.Arg, caps vllm.Caps, present string) (flags []string, notes []string) {
	asked := map[string]bool{}
	if split, err := config.SplitArgs(present); err == nil {
		for _, a := range split {
			name, _, _ := strings.Cut(a, "=")
			asked[name] = true
		}
	}
	for _, a := range args {
		if asked[a.Flag] || !caps.HasFlag(a.Flag) {
			continue
		}
		added := []string{a.Flag}
		if len(a.Values) > 0 {
			value := ""
			for _, v := range a.Values {
				if vllm.Accepts(caps, a.Flag, v) {
					value = v
					break
				}
			}
			if value == "" {
				continue
			}
			added = append(added, value)
		}
		flags = append(flags, added...)
		said := strings.Join(added, " ")
		for _, w := range a.With {
			if !asked[w] && caps.HasFlag(w) {
				flags = append(flags, w)
				said += " and " + w
			}
		}
		asked[a.Flag] = true
		notes = append(notes, "Added "+said+". "+a.Why)
	}
	return flags, notes
}

// modelTypeFiles fetches whatever the model type's files are missing and
// returns the profile lines that point at them, with a note for anything
// fetched or not fetched.
func modelTypeFiles(dataDir string, sets []modeltype.FileSet) (env []string, notes []string) {
	for _, set := range sets {
		dir := filepath.Join(dataDir, set.Dir)
		if set.Env != "" {
			env = append(env, set.Env+"="+dir)
		}
		var fetched []string
		for _, name := range set.Names {
			path := filepath.Join(dir, name)
			if st, err := os.Stat(path); err == nil && st.Size() > 0 {
				continue
			}
			if err := download(set.BaseURL+name, path); err != nil {
				notes = append(notes, fmt.Sprintf("Fetching %s into %s failed (%v). %s", name, dir, err, set.Why))
				continue
			}
			fetched = append(fetched, name)
		}
		if len(fetched) > 0 {
			notes = append(notes, fmt.Sprintf("Fetched %s into %s. %s", strings.Join(fetched, ", "), dir, set.Why))
		}
	}
	return env, notes
}

// fixChatTemplate applies the first of fixes that matches the repo's bundled
// chat template, and returns the patched template.
func fixChatTemplate(ctx context.Context, repo string, fixes []modeltype.TemplateFix) (patched string, why string, err error) {
	if len(fixes) == 0 {
		return "", "", nil
	}
	text, found, err := hf.ChatTemplate(ctx, repo)
	if err != nil || !found {
		return "", "", err
	}
	for _, f := range fixes {
		out, ok, err := f.Apply(text)
		if err != nil {
			return "", "", err
		}
		if ok {
			return out, f.Why, nil
		}
	}
	return "", "", nil
}

func download(url, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	resp, err := httpClient.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	// Written beside the target and renamed, so an interrupted fetch cannot
	// leave a half file that later looks present.
	tmp, err := os.CreateTemp(filepath.Dir(path), ".fetch-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
