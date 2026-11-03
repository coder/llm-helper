package recipe

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/coder/vllm-helper/internal/hardware"
	"github.com/coder/vllm-helper/internal/provider"
)

//go:embed tested.json
var testedJSON []byte

// SchemaVersion is the recipes format this build reads, in tested.json and
// in a recipes.d file alike.
const SchemaVersion = 1

// Tested holds the recipes we have actually run on our own hardware. Entries
// are hand-authored from a real deployment: a configuration nobody has run is
// not a recipe, so nothing generates them.
type Tested struct {
	SchemaVersion int            `json:"schema_version"`
	Recipes       []TestedRecipe `json:"recipes"`
}

// TestedRecipe is one launch configuration proven on real hardware. Image is
// pinned by digest, never a moving tag: eugr's :latest advanced under us mid-
// session, and a moved image is how a valid backend name becomes invalid.
//
// A recipe is data imported from an upstream, never a script run from one:
// the image, the pinned models, the args and the env are all here, and the
// engine named by Provider says how they become a command line.
type TestedRecipe struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	// Provider is the engine id that runs the recipe; empty means vllm.
	Provider string `json:"provider,omitempty"`
	// Model is the name the API serves the model under, which is what the
	// profile and Coder name. For vLLM it is also the repo it loads.
	Model    string `json:"model"`
	Topology string `json:"topology"` // solo, shard or pool
	Nodes    int    `json:"nodes"`
	Image    string `json:"image"`
	Port     int    `json:"port"`
	// Requires is the hardware definition id the recipe runs on. Empty
	// falls back to the first word of verified.hardware.
	Requires string `json:"requires,omitempty"`
	// Models are the repos the recipe loads, pinned to revisions. The
	// engine's templates reach them as {weights_path}, {drafter_path} and
	// {adapter_path}.
	Models []Model `json:"models,omitempty"`
	// Params set the engine's own params by name.
	Params   map[string]string `json:"params,omitempty"`
	Source   Source            `json:"source"`
	Args     []string          `json:"args"`
	Env      map[string]string `json:"env,omitempty"`
	Verified Verification      `json:"verified"`

	// Local is the recipes.d file a local recipe came from, and "" for one
	// shipped in the binary.
	Local string `json:"-"`
}

// Model is one repo a recipe loads, pinned to a revision.
type Model struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
	// Role is weights, drafter or adapter.
	Role string `json:"role"`
	// Staging says how the revision reaches each node. "download" means
	// each node downloads it into its own cache before the launch, and the
	// launch only checks it is there.
	Staging string `json:"staging"`
}

// Source is the upstream recipe an entry was derived from, pinned to a commit.
// It is structured rather than a comment because JSON has no comments and, more
// importantly, "recipediff" needs it to know what to compare an entry against.
// It also carries the attribution MIT and Apache-2.0 upstreams require.
type Source struct {
	Repo    string `json:"repo"`
	Path    string `json:"path,omitempty"`
	Commit  string `json:"commit"`
	License string `json:"license,omitempty"`
	// Format is the layout of the upstream file, which picks the reader
	// recipediff compares it with.
	Format     string      `json:"format"`
	Deviations []Deviation `json:"deviations,omitempty"`
}

// Source formats. Each one but manual has a reader in tools/recipediff, so a
// new upstream layout is one reader and a fixture.
const (
	// FormatEugr is eugr/spark-vllm-docker's recipe YAML.
	FormatEugr = "eugr-recipe"
	// FormatEnvSh is a shell config file of KEY="${KEY:-default}" lines,
	// such as a TensorFold launcher's scripts/config.sh.
	FormatEnvSh = "env-sh"
	// FormatManual is an upstream with nothing machine-readable to compare
	// against. recipediff reports such an entry as unchecked.
	FormatManual = "manual"
)

var formats = []string{FormatEugr, FormatEnvSh, FormatManual}

// Deviation is one place an entry deliberately departs from its source. It is
// structured because prose cannot be compared: "recipes diff" has to tell a
// decision we made from drift nobody noticed, and a reader has to see what was
// changed without trusting a sentence to still be true.
type Deviation struct {
	Kind     string `json:"kind"`               // flag, env, mod or config: where the setting lives
	Setting  string `json:"setting"`            // what upstream calls it
	Upstream string `json:"upstream,omitempty"` // upstream's value, empty when it is ours alone
	Ours     string `json:"ours,omitempty"`     // ours, empty when we drop it
	Why      string `json:"why"`
}

// Deviation kinds. A flag is named the way upstream names it in defaults
// (max_model_len, not --max-model-len), an env key is the raw key, and a mod
// is the path upstream lists. Without the kind, a checker has to guess, and
// the guess that was there turned VLLM_MOE_SKIP_PADDING into the flag
// --VLLM-MOE-SKIP-PADDING and then failed to find it.
//
// A config deviation is a key in an env-sh upstream's config file, named as
// that file names it: such a file sets most flags through its own keys.
const (
	DeviationFlag   = "flag"
	DeviationEnv    = "env"
	DeviationMod    = "mod"
	DeviationConfig = "config"
)

// Change renders the move for display. An absent side is a drop or an
// addition, which reads better than an empty column.
func (d Deviation) Change() string {
	switch {
	case d.Upstream == "" && d.Ours == "":
		return "not used here"
	case d.Ours == "":
		return d.Upstream + " -> not used"
	case d.Upstream == "":
		return "added: " + d.Ours
	default:
		return d.Upstream + " -> " + d.Ours
	}
}

// Describe renders the provenance for "recipes show".
func (s Source) Describe() string {
	if s.Repo == "" {
		return ""
	}
	out := s.Repo
	if s.Path != "" {
		out += " " + s.Path
	}
	if len(s.Commit) >= 8 {
		out += " @" + s.Commit[:8]
	}
	if s.License != "" {
		out += " (" + s.License + ")"
	}
	return out
}

// Verification is what separates our entries from an upstream claim: upstream
// says a model works somewhere, this says it ran here, on this date, at this
// image digest, with these numbers.
type Verification struct {
	Date        string   `json:"date"`
	Hardware    string   `json:"hardware"`
	Driver      string   `json:"driver,omitempty"`
	VLLMVersion string   `json:"vllm_version,omitempty"`
	LoadSeconds int      `json:"load_seconds,omitempty"`
	Context     int      `json:"context,omitempty"`
	KVTokens    int64    `json:"kv_tokens,omitempty"`
	Concurrency float64  `json:"concurrency,omitempty"`
	Smoke       string   `json:"smoke,omitempty"`
	Notes       []string `json:"notes,omitempty"`
}

// LoadTested returns the recipes in effect: the shipped ones, plus any a
// recipes.d file added; see UseLocal.
func LoadTested() (Tested, error) {
	shipped, err := parseShipped()
	if err != nil {
		return Tested{}, err
	}
	localMu.Lock()
	local := slices.Clone(localRecipes)
	localMu.Unlock()
	shipped.Recipes = append(shipped.Recipes, local...)
	sort.Slice(shipped.Recipes, func(i, j int) bool { return shipped.Recipes[i].ID < shipped.Recipes[j].ID })
	return shipped, nil
}

func parseShipped() (Tested, error) {
	t, err := decode(testedJSON)
	if err != nil {
		return Tested{}, fmt.Errorf("parse tested recipes: %w", err)
	}
	var errs []error
	for _, r := range t.Recipes {
		if err := r.validate(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := errors.Join(errs...); err != nil {
		return Tested{}, fmt.Errorf("invalid tested recipes: %w", err)
	}
	return t, nil
}

func decode(data []byte) (Tested, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var t Tested
	if err := dec.Decode(&t); err != nil {
		return Tested{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return Tested{}, errors.New("unexpected data after the top-level object")
	}
	if t.SchemaVersion != SchemaVersion {
		return Tested{}, fmt.Errorf("schema_version is %d; this llm-helper reads version %d", t.SchemaVersion, SchemaVersion)
	}
	return t, nil
}

var (
	recipeIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*$`)
	revisionPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// validate holds a recipe to what a launch relies on, so a mistake fails
// when the recipe loads rather than halfway through starting a cluster.
func (r TestedRecipe) validate() error {
	var errs []error
	add := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	if !recipeIDPattern.MatchString(r.ID) {
		add("id must be lowercase letters, digits, dots and dashes")
	}
	if r.DisplayName == "" || r.Model == "" {
		add("display_name and model are required")
	}
	if r.Topology != "solo" && r.Topology != "shard" && r.Topology != "pool" {
		add("topology %q must be solo, shard or pool", r.Topology)
	}
	if r.Nodes < 1 || r.Port < 1 || r.Port > 65535 {
		add("nodes must be at least 1 and port a TCP port")
	}
	if r.Image != "" && !strings.Contains(r.Image, "@sha256:") {
		add("image %q is not pinned by digest", r.Image)
	}
	if !slices.Contains(formats, r.Source.Format) {
		add("source.format %q must be one of %s", r.Source.Format, strings.Join(formats, ", "))
	}
	if r.Requires != "" {
		if _, ok := hardware.Default().ByID(r.Requires); !ok {
			add("requires %q names no hardware definition", r.Requires)
		}
	}
	e, err := provider.LookupEngine(r.Provider)
	if err != nil {
		add("%v", err)
	} else if !e.Has(provider.RuntimeContainer) {
		add("engine %s has no container runtime, so it cannot launch a recipe", e.ID)
	} else {
		for name := range r.Params {
			if !slices.ContainsFunc(e.Params, func(p provider.Param) bool { return p.Name == name }) {
				add("params sets %q, which engine %s does not declare", name, e.ID)
			}
		}
		if bad := e.UnknownPlaceholders(r.Args); len(bad) > 0 {
			add("args use {%s}, which engine %s cannot fill", strings.Join(bad, "}, {"), e.ID)
		}
		have := map[string]bool{}
		for _, m := range r.Models {
			have[m.Role] = true
		}
		for _, role := range append(e.UsesModelPaths(), provider.ModelRolesIn(r.Args)...) {
			if !have[role] {
				add("engine %s needs a pinned %s model, and models has none", e.ID, role)
			}
		}
		if len(r.Models) > 0 && e.HFCacheHostPath(r.Params) == "" {
			add("models are pinned, but engine %s mounts no Hugging Face cache to find them in", e.ID)
		}
	}
	roles := map[string]bool{}
	for _, m := range r.Models {
		switch {
		case m.ID == "" || !strings.Contains(m.ID, "/"):
			add("model %q is not an org/name repo id", m.ID)
		case !revisionPattern.MatchString(m.Revision):
			add("model %s: revision %q is not a full commit sha", m.ID, m.Revision)
		case !slices.Contains(provider.ModelRoles, m.Role):
			add("model %s: role %q must be one of %s", m.ID, m.Role, strings.Join(provider.ModelRoles, ", "))
		case roles[m.Role]:
			add("model %s: a recipe pins one model per role, and %s is taken", m.ID, m.Role)
		case m.Staging != "download":
			add("model %s: staging %q must be download", m.ID, m.Staging)
		}
		roles[m.Role] = true
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("recipe %q: %w", r.ID, err)
	}
	return nil
}

// Engine is the engine that runs the recipe. Recipes are validated when they
// load, so an unknown engine cannot get this far; the zero Engine it would
// return renders nothing.
func (r TestedRecipe) Engine() provider.Engine {
	e, _ := provider.LookupEngine(r.Provider)
	return e
}

// ModelRefs are the recipe's pinned models by role.
func (r TestedRecipe) ModelRefs() map[string]provider.ModelRef {
	out := map[string]provider.ModelRef{}
	for _, m := range r.Models {
		out[m.Role] = provider.ModelRef{ID: m.ID, Revision: m.Revision}
	}
	return out
}

// Get looks a recipe up by id.
func (t Tested) Get(id string) (TestedRecipe, bool) {
	for _, r := range t.Recipes {
		if r.ID == id {
			return r, true
		}
	}
	return TestedRecipe{}, false
}

// Summary is the one-line description shown next to the display name.
func (r TestedRecipe) Summary() string {
	out := r.nodeLabel()
	if r.Verified.Context > 0 {
		out += " - " + humanTokens(int64(r.Verified.Context)) + " ctx"
	}
	if r.Verified.Concurrency > 0 {
		out += fmt.Sprintf(" - %.1fx conc", r.Verified.Concurrency)
	}
	if r.Verified.LoadSeconds > 0 {
		out += fmt.Sprintf(" - ~%.1f min load", float64(r.Verified.LoadSeconds)/60)
	}
	return out
}

func (r TestedRecipe) nodeLabel() string {
	switch {
	case r.Nodes > 1 && r.Topology == "pool":
		return fmt.Sprintf("%d nodes, replicas", r.Nodes)
	case r.Nodes > 1:
		return fmt.Sprintf("%d nodes, sharded", r.Nodes)
	default:
		return "1 node"
	}
}

func humanTokens(n int64) string {
	switch {
	case n >= 1<<20 && n%(1<<20) == 0:
		return fmt.Sprintf("%dM", n/(1<<20))
	case n >= 1024 && n%1024 == 0:
		return fmt.Sprintf("%dK", n/1024)
	default:
		return fmt.Sprintf("%d", n)
	}
}
