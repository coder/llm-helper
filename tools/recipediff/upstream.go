package main

import "strings"

// upstream is one source format's reading of the file an entry cites.
type upstream interface {
	// mismatch says why the file cannot be the entry's source, or "".
	mismatch(e entry) string
	// findings are every place the entry disagrees with the file.
	findings(e entry) []finding
	// moved is what changed between this revision and a newer one.
	moved(newer upstream) []change
}

// reader parses one source format.
type reader func([]byte) (upstream, error)

// readers are keyed by a recipe's source.format. A new upstream layout is
// one reader here and a fixture; a format with none, such as manual, is
// reported as not checked.
var readers = map[string]reader{
	"eugr-recipe": func(b []byte) (upstream, error) {
		r, err := parseEugr(b)
		if err != nil {
			return nil, err
		}
		return r, nil
	},
	"env-sh": func(b []byte) (upstream, error) { return parseEnvSh(b), nil },
}

func (r eugrRecipe) mismatch(e entry) string {
	if r.Model != e.Model {
		return "that file serves " + r.Model
	}
	return ""
}

func (r eugrRecipe) findings(e entry) []finding { return compare(e, r) }

func (r eugrRecipe) moved(newer upstream) []change {
	n, ok := newer.(eugrRecipe)
	if !ok {
		return nil
	}
	return diffUpstream(r, n)
}

func (s envSh) mismatch(e entry) string {
	if missing := s.missingModels(e); len(missing) > 0 {
		return "that file does not pin " + strings.Join(missing, ", ")
	}
	return ""
}

func (s envSh) findings(e entry) []finding { return compareEnvSh(e, s) }

func (s envSh) moved(newer upstream) []change {
	n, ok := newer.(envSh)
	if !ok {
		return nil
	}
	return diffEnvSh(s, n)
}
