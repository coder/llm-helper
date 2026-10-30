package coder

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/plugin"
	"github.com/coder/vllm-helper/internal/profile"
	"github.com/coder/vllm-helper/internal/provider"
)

// Configuration keys. All three are required: a URL without a token, or
// either without the provider to hang the models off, can't sync anything.
const (
	keyURL        = "CODER_URL"
	keyToken      = "CODER_API_TOKEN"
	keyProviderID = "CODER_AI_PROVIDER_ID"
	// Optional. Coder caps a reply at its own default (32,000 tokens) unless
	// the model entry carries an override, and that cap is server-side, so a
	// deployment that wants longer answers has to say so here.
	keyMaxOutput = "CODER_MAX_OUTPUT_TOKENS"
)

const notConfigured = "Coder sync: not configured (" +
	keyURL + "/" + keyToken + "/" + keyProviderID + "), skipping."

const syncUsage = "usage: vllm-helper coder-sync <activate|register|remove> <profile>\n" +
	"       vllm-helper coder-sync register-all\n" +
	"       vllm-helper coder-sync prune [--yes]"

func init() { plugin.Register(Plugin{}) }

// Plugin mirrors local profiles into a Coder deployment's model picker.
type Plugin struct{}

func (Plugin) Name() string { return "coder" }

func (Plugin) Configured(cfg *config.Config) bool {
	return cfg != nil && cfg.HasAll(keyURL, keyToken, keyProviderID)
}

// Events: Coder needs to know which model is selectable, so it reacts to a
// switch and to a profile being deleted. It deliberately does not use
// EventReady -- see the comment on activate.
func (Plugin) Events() []plugin.Event {
	return []plugin.Event{plugin.EventActivating, plugin.EventRemoved}
}

func (p Plugin) Handle(_ context.Context, e plugin.Event, s plugin.State) error {
	switch e {
	case plugin.EventActivating:
		activate(s.Config, s.Profile)
	case plugin.EventRemoved:
		remove(s.Config, s.Name)
	default:
		// Only reachable if Events above grows an entry this switch
		// doesn't, which would otherwise be a silent no-op.
		return fmt.Errorf("no handler for event %q", e)
	}
	// Every failure inside is already reported where it happens, with the
	// context needed to act on it; returning it too would print it twice.
	return nil
}

func (p Plugin) Check(_ context.Context, cfg *config.Config) error {
	client := newClient(cfg)
	if client == nil {
		return nil
	}
	if err := client.Check(); err != nil {
		return err
	}
	return tokenExpiring(client)
}

// tokenExpiring reports a token close enough to expiry to act on.
//
// Coder caps token lifetime, so "expires eventually" is true for most of a
// token's life and warning on it would be noise. The last two days are worth
// interrupting for, because an expired token vanishes from the token list and
// the only symptom is that Coder quietly stops being told what is serving.
func tokenExpiring(client *Client) error {
	k, err := client.TokenExpiry()
	if err != nil || k.ExpiresAt.IsZero() {
		// The sync itself works, so an unreadable key is not worth failing a
		// health check over.
		return nil
	}
	left := time.Until(k.ExpiresAt)
	switch {
	case left <= 0:
		return fmt.Errorf("%s expired on %s -- mint a new one and set it in secrets.env",
			keyToken, k.ExpiresAt.Format(time.DateOnly))
	case left < 48*time.Hour:
		return fmt.Errorf("%s expires in %dh (on %s) -- mint a new one and set it in secrets.env",
			keyToken, int(left.Hours()), k.ExpiresAt.Format(time.DateOnly))
	}
	return nil
}

func (p Plugin) Commands() []*cobra.Command {
	c := &cobra.Command{
		Use:   "coder-sync <activate|register|remove|register-all|prune> [profile]",
		Short: "Manually sync a profile's Coder model entry",
		// GroupID is left to the CLI, which owns how help is organised.
		Example: "  vllm-helper coder-sync register qwen\n" +
			"  vllm-helper coder-sync register-all\n" +
			"  vllm-helper coder-sync prune",
		Args: cobra.MaximumNArgs(2),
		RunE: func(c *cobra.Command, args []string) error {
			yes, _ := c.Flags().GetBool("yes")
			return run(args, yes)
		},
	}
	c.Flags().Bool("yes", false, "delete pruned entries without asking")
	return []*cobra.Command{c}
}

// prune deletes model entries on the configured provider that no local profile
// accounts for.
//
// The provider names one endpoint, which is this machine, so an entry naming a
// model this machine cannot serve is one the picker offers and the server then
// refuses. They accumulate: a profile deleted outside vllm-helper, or -- as
// happened here -- a provider whose address moved to a different box, leaving
// the previous machine's models behind.
//
// Reports by default and only deletes when asked, since the entries are not
// ours to assume about.
func prune(cfg *config.Config, yes bool) error {
	client := newClient(cfg)
	if client == nil {
		return errors.New(notConfigured)
	}
	entries, err := client.List()
	if err != nil {
		return err
	}
	profiles, err := profile.List(cfg.Dirs)
	if err != nil {
		return err
	}
	local := map[string]bool{}
	for _, p := range profiles {
		local[p.Model()] = true
	}

	var orphans []Entry
	for _, e := range entries {
		if !local[e.Model] {
			orphans = append(orphans, e)
		}
	}
	reportDuplicates(entries)
	if len(orphans) == 0 {
		fmt.Printf("Coder sync: all %d entries on this provider match a local profile.\n", len(entries))
		return nil
	}

	fmt.Printf("%d entr%s on this provider name a model this machine does not serve:\n",
		len(orphans), plural(len(orphans), "y", "ies"))
	for _, o := range orphans {
		state := ""
		if o.Enabled {
			// Worth calling out: a disabled entry is merely untidy, an
			// enabled one is offered in the picker and then fails.
			state = "  [enabled -- selectable, and will fail]"
		}
		fmt.Printf("  %s%s\n", o.Model, state)
	}
	if !yes && !confirmDelete() {
		fmt.Println("Left alone. Re-run with --yes to delete them.")
		return nil
	}
	for _, o := range orphans {
		if err := client.Delete(o.ID); err != nil && !errors.Is(err, ErrNotFound) {
			fmt.Fprintf(os.Stderr, "  failed to delete %s: %v\n", o.Model, err)
			continue
		}
		fmt.Printf("  deleted %s\n", o.Model)
	}
	return nil
}

// reportDuplicates names models with more than one entry on this provider.
//
// Only reported, never deleted: the rows are indistinguishable from here, and
// the one to keep may be the one an operator gave options to in the UI.
// Syncing now adopts by model, so new duplicates are not created; these are
// left over from when state was keyed by profile name alone.
func reportDuplicates(entries []Entry) {
	for _, dups := range duplicateModels(entries) {
		fmt.Printf("%d entries name %s; the picker will show it more than once:\n", len(dups), dups[0].Model)
		for _, d := range dups {
			state := "disabled"
			if d.Enabled {
				state = "enabled"
			}
			fmt.Printf("  %s  [%s]\n", d.ID, state)
		}
		fmt.Println("  Delete the ones you do not want in Coder; this command will not choose for you.")
	}
}

// duplicateModels groups entries that name the same model, in the order the
// models first appear so the report is stable.
func duplicateModels(entries []Entry) [][]Entry {
	byModel := map[string][]Entry{}
	var order []string
	for _, e := range entries {
		if _, seen := byModel[e.Model]; !seen {
			order = append(order, e.Model)
		}
		byModel[e.Model] = append(byModel[e.Model], e)
	}
	var out [][]Entry
	for _, model := range order {
		if len(byModel[model]) > 1 {
			out = append(out, byModel[model])
		}
	}
	return out
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

// confirmDelete asks before removing entries. The CLI has its own prompt, but
// a plugin importing the package that loads it would be a cycle -- and this is
// the only place here that asks anything.
func confirmDelete() bool {
	fmt.Print("Delete them? [y/N] ")
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true
	}
	return false
}

func run(args []string, yes bool) error {
	if len(args) < 1 {
		return errors.New(syncUsage)
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	op := args[0]

	if op == "register-all" {
		return registerAll(cfg)
	}
	if op == "prune" {
		return prune(cfg, yes)
	}
	if len(args) < 2 {
		return errors.New(syncUsage)
	}
	name := args[1]
	p, err := profile.Load(cfg.Dirs, name)
	// A profile that won't load is fatal except for remove, so a Coder
	// entry can still be cleaned up after its profile file is gone.
	if err != nil && op != "remove" {
		return err
	}
	switch op {
	case "activate":
		activate(cfg, p)
	case "register":
		register(cfg, p)
	case "remove":
		remove(cfg, name)
	default:
		return errors.New(syncUsage)
	}
	return nil
}

// registerAll backfills a (disabled) Coder model-config entry for every
// saved profile, so the deployment's model picker shows the whole library,
// not just models that have been active at least once.
func registerAll(cfg *config.Config) error {
	if !(Plugin{}).Configured(cfg) {
		return errors.New("Coder sync is not configured (" +
			keyURL + "/" + keyToken + "/" + keyProviderID + ")")
	}
	profiles, err := profile.List(cfg.Dirs)
	if err != nil {
		return err
	}
	if len(profiles) == 0 {
		fmt.Println("No profiles to register.")
		return nil
	}
	for _, p := range profiles {
		register(cfg, p)
	}
	fmt.Printf("Done -- %d profile(s) checked against Coder.\n", len(profiles))
	return nil
}

// ids loads the profile -> Coder model-config-id state file.
func ids(cfg *config.Config) *config.EnvFile {
	f, err := config.ParseEnvFile(cfg.Dirs.PluginStateFile("coder"))
	if err != nil {
		return &config.EnvFile{}
	}
	return f
}

func saveIDs(cfg *config.Config, f *config.EnvFile) {
	_ = os.MkdirAll(cfg.Dirs.State, 0o755)
	if err := f.Write(cfg.Dirs.PluginStateFile("coder"), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "Coder sync: couldn't save state:", err)
	}
}

func newClient(cfg *config.Config) *Client {
	if !(Plugin{}).Configured(cfg) {
		return nil
	}
	return NewClient(cfg.Lookup(keyURL, ""), cfg.Lookup(keyToken, ""),
		cfg.Lookup(keyProviderID, ""), cfg.CoderOrganization())
}

// modelConfig translates a profile into the entry Coder stores.
//
// Coder rejects a context_limit of 0, so an unparseable MAX_MODEL_LEN has to
// be reported here: sending it produced an HTTP 400 whose message said
// nothing about which local setting was at fault.
//
// A profile for a server other than vLLM leaves the context to that server and
// so has no MAX_MODEL_LEN. Coder is then told DefaultMaxModelLen, the context a
// profile gets when nothing says, and setting MAX_MODEL_LEN corrects it.
func modelConfig(cfg *config.Config, p *profile.Profile, enabled bool) (ModelConfig, error) {
	raw := p.MaxModelLen()
	if raw == "" {
		// A provider that sizes its own context leaves no MAX_MODEL_LEN, and
		// failing sync for that would leave the previous model enabled after a
		// switch. An unknown PROVIDER gets no such grace: it cannot be served,
		// and inventing a context limit would hide that.
		if prov, err := provider.Lookup(p.Provider()); err == nil && !prov.Capabilities().VLLMArgs {
			raw = strconv.Itoa(profile.DefaultMaxModelLen)
		}
	}
	limit, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || limit <= 0 {
		return ModelConfig{}, fmt.Errorf(
			"MAX_MODEL_LEN is %q, which is not a context length Coder will accept -- fix it with 'vllm-helper set %s MAX_MODEL_LEN=<number>'",
			p.MaxModelLen(), p.Name)
	}
	return ModelConfig{
		Model:        p.Model(),
		DisplayName:  p.DisplayName(),
		ContextLimit: limit,
		Enabled:      enabled,
		IsDefault:    enabled,
		Options:      outputOptions(cfg),
	}, nil
}

// outputOptions is the per-model override Coder stores alongside the entry.
//
// Unset means send nothing, which leaves whatever is configured in Coder
// alone: an empty object would overwrite an operator's setting with nothing.
func outputOptions(cfg *config.Config) map[string]any {
	if cfg == nil {
		return nil
	}
	raw := strings.TrimSpace(cfg.Lookup(keyMaxOutput, ""))
	if raw == "" {
		return nil
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n <= 0 {
		fmt.Fprintf(os.Stderr, "Coder sync: %s is %q, which is not a token count; ignoring it\n", keyMaxOutput, raw)
		return nil
	}
	return map[string]any{"max_output_tokens": n}
}

// activate makes this profile the single enabled/default model in Coder and
// disables every other known profile's entry. Failures warn and continue --
// Coder sync never blocks a local switch.
//
// This runs before the service restarts, so Coder briefly advertises a model
// that is still loading. Deferring it until the endpoint is healthy would
// need a hook that nothing raises under --no-follow, and Coder retries an
// unreachable provider for long enough to cover a normal load, so the entry
// goes in early and stays put.
//
// Disabling the others is not about is_default (Coder enforces a single
// default itself); it is because vLLM serves one model at a time, so any
// other entry names something this machine cannot answer for right now.
func activate(cfg *config.Config, p *profile.Profile) {
	client := newClient(cfg)
	if client == nil {
		fmt.Println(notConfigured)
		return
	}
	mc, err := modelConfig(cfg, p, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Coder sync:", err)
		return
	}
	state := ids(cfg)

	// A saved ID can name an entry that no longer exists. Recreating is what
	// was wanted; refusing would leave the picker permanently stale.
	id := state.Get(p.Name)
	if id == "" {
		// Nothing saved does not mean nothing exists. State is per profile
		// name, but the picker is per model, so a renamed profile or a second
		// machine syncing the same model would otherwise post a duplicate row
		// and then fight with it over enabled/is_default.
		if adopted, err := client.FindByModel(mc.Model); err != nil {
			fmt.Fprintln(os.Stderr, "Coder sync: could not look for an existing entry:", err)
		} else if adopted != "" {
			fmt.Printf("Coder sync: adopting the existing entry for %s.\n", mc.Model)
			id = adopted
			state.Set(p.Name, id)
			saveIDs(cfg, state)
		}
	}
	if id != "" {
		fmt.Printf("Coder sync: updating model-config for '%s'...\n", p.Name)
		err := client.Update(id, mc)
		if err == nil {
			state = disableOthers(cfg, client, state, p.Name)
			fmt.Printf("Coder sync: '%s' is now the only enabled/default model in Coder Agents.\n", p.Name)
			return
		}
		if !errors.Is(err, ErrNotFound) {
			fmt.Fprintln(os.Stderr, "Coder sync: update failed, continuing without it:", err)
			return
		}
		fmt.Printf("Coder sync: '%s' had no entry on this deployment, creating one...\n", p.Name)
	} else {
		fmt.Printf("Coder sync: creating model-config for '%s'...\n", p.Name)
	}
	newID, err := client.Create(mc)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Coder sync: create failed, continuing without it:", err)
		return
	}
	state.Set(p.Name, newID)
	saveIDs(cfg, state)

	state = disableOthers(cfg, client, state, p.Name)
	fmt.Printf("Coder sync: '%s' is now the only enabled/default model in Coder Agents.\n", p.Name)
}

// disableOthers turns off every other profile's entry: vLLM serves one model
// at a time, so any other entry names something this machine cannot answer
// for. An entry that has already gone is forgotten rather than reported.
//
// Entries are owned by ID, not by profile name: adoption maps a second
// profile for the same model onto the existing row, and disabling that row on
// behalf of the other name turned off the entry this switch just enabled.
func disableOthers(cfg *config.Config, client *Client, state *config.EnvFile, keep string) *config.EnvFile {
	profiles, _ := profile.List(cfg.Dirs)
	keepID := state.Get(keep)
	dropped := false
	for _, other := range profiles {
		if other.Name == keep {
			continue
		}
		id := state.Get(other.Name)
		if id == "" || id == keepID {
			continue
		}
		switch err := client.Disable(id); {
		case err == nil:
		case errors.Is(err, ErrNotFound):
			state.Delete(other.Name)
			dropped = true
		default:
			fmt.Fprintf(os.Stderr, "Coder sync: failed to disable '%s', continuing: %v\n", other.Name, err)
		}
	}
	if dropped {
		saveIDs(cfg, state)
	}
	return state
}

// register creates a disabled entry for a profile that isn't active
// (backfill), without touching anything else's state.
func register(cfg *config.Config, p *profile.Profile) {
	client := newClient(cfg)
	if client == nil {
		fmt.Println(notConfigured)
		return
	}
	state := ids(cfg)
	if state.Get(p.Name) != "" {
		fmt.Printf("Coder sync: '%s' already has a model-config, leaving as-is.\n", p.Name)
		return
	}
	mc, err := modelConfig(cfg, p, false)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Coder sync:", err)
		return
	}
	fmt.Printf("Coder sync: registering (disabled) model-config for '%s'...\n", p.Name)
	newID, err := client.Create(mc)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Coder sync: register failed, continuing without it:", err)
		return
	}
	state.Set(p.Name, newID)
	saveIDs(cfg, state)
}

// remove deletes a profile's Coder entry (used by `remove`). An entry another
// profile also maps to is still in use, so only this name's claim on it goes.
func remove(cfg *config.Config, name string) {
	client := newClient(cfg)
	if client == nil {
		fmt.Println(notConfigured)
		return
	}
	state := ids(cfg)
	id := state.Get(name)
	if id == "" {
		return
	}
	if other := sharedWith(state, id, name); other != "" {
		fmt.Printf("Coder sync: keeping the model-config '%s' shares with '%s'.\n", name, other)
	} else {
		fmt.Printf("Coder sync: deleting model-config for '%s'...\n", name)
		if err := client.Delete(id); err != nil {
			fmt.Fprintln(os.Stderr, "Coder sync: delete failed (may already be gone), continuing:", err)
		}
	}
	state.Delete(name)
	saveIDs(cfg, state)
}

// sharedWith names another profile mapped to the same entry, or "".
func sharedWith(state *config.EnvFile, id, except string) string {
	for _, k := range state.Keys() {
		if k != except && state.Get(k) == id {
			return k
		}
	}
	return ""
}
