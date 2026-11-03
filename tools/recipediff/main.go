// Command recipediff reports where our tested recipes disagree with the
// upstream recipes they were derived from.
//
// It is a development tool, not part of the CLI: it never ships in the binary
// and cannot break a launch. That is what lets it depend on a YAML parser and
// on the network, and what lets it tolerate a partial parse -- we compare
// against an upstream's file, we never execute it.
//
// Each source format an entry can cite has a reader here: eugr-recipe for
// eugr/spark-vllm-docker's recipe YAML and env-sh for a shell config file of
// KEY="${KEY:-default}" lines. An entry whose format has no reader, such as
// manual, is reported as not checked and counted in the exit code.
//
// It reports disagreement and never resolves it. A difference we meant to make
// belongs in the entry's source.deviations; anything else is a difference
// nobody has looked at.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

func main() {
	recipes := flag.String("recipes", filepath.Join("..", "..", "recipes"), "directory of tested recipes, one JSON file each")
	dir := flag.String("upstream-dir", "", "read upstream recipes from this directory instead of fetching them")
	flag.Parse()

	n, err := run(*recipes, *dir, os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "recipediff:", err)
		os.Exit(2)
	}
	if n > 0 {
		os.Exit(1)
	}
}

func run(recipesDir, dir string, w io.Writer) (int, error) {
	t, err := readTested(recipesDir)
	if err != nil {
		return 0, err
	}

	total, unchecked := 0, 0
	for _, e := range t.Recipes {
		read, ok := readers[e.Source.Format]
		if !ok {
			// Counted, not waved through: an entry nobody can compare is an
			// entry whose differences nobody has seen. Skipping it silently
			// let the tool exit 0 while saying so on the line above.
			fmt.Fprintf(w, "%s: NOT CHECKED -- no reader for source format %q (%s)\n\n", e.ID, e.Source.Format, e.Source.Repo)
			unchecked++
			continue
		}
		body, err := upstreamBytes(e, dir)
		if err != nil {
			return 0, fmt.Errorf("%s: %w", e.ID, err)
		}
		up, err := read(body)
		if err != nil {
			return 0, fmt.Errorf("%s: %w", e.ID, err)
		}
		if why := up.mismatch(e); why != "" {
			fmt.Fprintf(w, "%s: WRONG SOURCE -- %s\n\n", e.ID, why)
			total++
			continue
		}
		fs := up.findings(e)
		total += undeclared(fs)
		report(w, e, fs, dir, up, read)
	}
	fmt.Fprintf(w, "%d undeclared difference(s) and %d unchecked entr(ies) across %d entries\n", total, unchecked, len(t.Recipes))
	return total + unchecked, nil
}

// readTested reads every recipe file in dir, in name order.
func readTested(dir string) (tested, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return tested{}, err
	}
	if len(paths) == 0 {
		return tested{}, fmt.Errorf("no recipes in %s", dir)
	}
	var t tested
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return tested{}, err
		}
		var e entry
		if err := json.Unmarshal(raw, &e); err != nil {
			return tested{}, fmt.Errorf("parse %s: %w", p, err)
		}
		t.Recipes = append(t.Recipes, e)
	}
	return t, nil
}

// reportStaleness says whether the pin still describes upstream. A commit
// touching the file is not news by itself; what the pin has missed is.
func reportStaleness(w io.Writer, e entry, pinned upstream, read reader) {
	head, err := headCommit(e.Source.Repo, e.Source.Path)
	if err != nil {
		fmt.Fprintf(w, "  staleness unknown: %v\n", err)
		return
	}
	if head == e.Source.Commit {
		fmt.Fprintln(w, "  pinned at the newest commit for this file")
		return
	}
	body, err := get(fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s", e.Source.Repo, head, e.Source.Path))
	if err != nil {
		fmt.Fprintf(w, "  pin trails %s, could not read it: %v\n", short(head), err)
		return
	}
	newest, err := read(body)
	if err != nil {
		fmt.Fprintf(w, "  pin trails %s, could not parse it: %v\n", short(head), err)
		return
	}
	changes := pinned.moved(newest)
	if len(changes) == 0 {
		fmt.Fprintf(w, "  pin trails %s, but the file is unchanged\n", short(head))
		return
	}
	fmt.Fprintf(w, "  UPSTREAM MOVED since %s, newest %s\n", short(e.Source.Commit), short(head))
	for _, c := range changes {
		fmt.Fprintf(w, "    %-6s %-40s %s -> %s\n", c.Kind, c.Key, quote(c.Was), quote(c.Now))
	}
}

func report(w io.Writer, e entry, fs []finding, dir string, pinned upstream, read reader) {
	fmt.Fprintf(w, "%s  (%s %s @%s)\n", e.ID, e.Source.Repo, e.Source.Path, short(e.Source.Commit))
	if dir == "" {
		reportStaleness(w, e, pinned, read)
	}
	section := func(title string, want func(finding) bool) {
		var rows []finding
		for _, f := range fs {
			if want(f) {
				rows = append(rows, f)
			}
		}
		if len(rows) == 0 {
			return
		}
		fmt.Fprintf(w, "  %s\n", title)
		for _, f := range rows {
			fmt.Fprintf(w, "    %-6s %-40s %s\n", f.Kind, f.Label, move(f))
		}
	}
	section("declared", func(f finding) bool { return f.Declared })
	section("MISDECLARED -- the declaration describes a different move", func(f finding) bool { return f.Misdeclared })
	section("UNDECLARED", func(f finding) bool { return !f.Declared && !f.Misdeclared && !f.Info })
	section("ours alone, not a conflict", func(f finding) bool { return f.Info })
	fmt.Fprintln(w)
}

func move(f finding) string {
	switch {
	case f.Ours == "" && f.Upstream == "":
		return "present in both, empty"
	case f.Ours == "":
		return quote(f.Upstream) + " -> not used"
	case f.Upstream == "":
		return "only ours: " + quote(f.Ours)
	default:
		return quote(f.Upstream) + " -> " + quote(f.Ours)
	}
}

func quote(s string) string {
	if len(s) > 48 {
		return s[:45] + "..."
	}
	if s == "" {
		return "(flag only)"
	}
	return s
}

func short(sha string) string {
	if len(sha) >= 8 {
		return sha[:8]
	}
	return sha
}

func upstreamBytes(e entry, dir string) ([]byte, error) {
	if dir != "" {
		return os.ReadFile(filepath.Join(dir, filepath.Base(e.Source.Path)))
	}
	url := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s", e.Source.Repo, e.Source.Commit, e.Source.Path)
	return get(url)
}

// headCommit is the newest commit touching the file we pinned. A pin that
// trails it is the staleness signal; we ran DeepSeek for a week without
// speculative decoding because nothing watched for this.
func headCommit(repo, path string) (string, error) {
	url := fmt.Sprintf("https://api.github.com/repos/%s/commits?path=%s&per_page=1", repo, path)
	body, err := get(url)
	if err != nil {
		return "", err
	}
	var commits []struct {
		SHA string `json:"sha"`
	}
	if err := json.Unmarshal(body, &commits); err != nil {
		return "", err
	}
	if len(commits) == 0 {
		return "", fmt.Errorf("no commits reported for %s", path)
	}
	return commits[0].SHA, nil
}

func get(url string) ([]byte, error) {
	c := &http.Client{Timeout: 20 * time.Second}
	resp, err := c.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: http %d", url, resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 1<<20))
}
