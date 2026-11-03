package provider

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"regexp"
	"slices"
	"strings"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/hardware"
)

//go:embed engines.json
var enginesJSON []byte

// SchemaVersion is the engine definitions format this build reads. A file
// written for another version is refused rather than half understood.
const SchemaVersion = 1

// Runtimes an engine can declare.
const (
	// RuntimeProcess runs the server on the host, under the service unit.
	RuntimeProcess = "process"
	// RuntimeContainer runs it in a container on each node a recipe names.
	RuntimeContainer = "container"
)

// Engine is one inference server, defined as data. Its id is the PROVIDER
// value that selects it, so a config written before engines were data keeps
// working.
//
// Everything an engine needs to be launched is here: the argv and env
// templates, how it takes the model and the API key, where it binds and how
// its health is read. Behaviour that data cannot express, such as vLLM's
// memory fit, is a Go hook gated by a capability, and shared code acts on
// the capability, never on an engine's name.
type Engine struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	Why         string `json:"why"`
	// Runtime lists where the engine can run: on the host as a process, in
	// a container, or both.
	Runtime []string `json:"runtime"`
	// GPUVendors are the hardware vendor ids it runs on, plus "cpu"; empty
	// means it makes no claim.
	GPUVendors []string `json:"gpu_vendors,omitempty"`
	// Install says how to get the server, keyed by GOOS or "any".
	Install map[string]string `json:"install,omitempty"`
	// Params are engine-specific settings a recipe can set by name, each
	// with a default and a reason.
	Params []Param `json:"params,omitempty"`

	// Bin, Serve, OptionalArgs, Env, ModelArg, Bind and APIKey describe the
	// process runtime.
	Bin *Bin `json:"bin,omitempty"`
	// Serve is the argv template. It starts with {bin}.
	Serve []string `json:"serve,omitempty"`
	// OptionalArgs are argv groups appended after Serve, each dropped when
	// any placeholder in it is empty.
	OptionalArgs [][]string `json:"optional_args,omitempty"`
	// Env is added to the server's environment; an entry that renders empty
	// is left out.
	EnvVars  map[string]string `json:"env,omitempty"`
	ModelArg *ModelArg         `json:"model_arg,omitempty"`
	Bind     *Bind             `json:"bind,omitempty"`
	// APIKey is how the server takes its key: "flag:--api-key" puts it on
	// argv through {api_key}, "env:NAME" in the environment, and "none" means
	// the server checks no key at all.
	APIKey string `json:"api_key"`

	Health Health `json:"health"`
	// Container describes the container runtime, which recipes launch.
	Container *Container   `json:"container,omitempty"`
	Caps      Capabilities `json:"capabilities"`

	// Source is the engines.d file a local definition came from, and "" for
	// one shipped in the binary.
	Source string `json:"-"`
}

// Param is a named engine setting that templates reference as {name}.
type Param struct {
	Name    string `json:"name"`
	Default string `json:"default"`
	Why     string `json:"why"`
}

// Bin is the server executable for the process runtime.
type Bin struct {
	// Key is the config.env setting holding its full path.
	Key string `json:"key"`
	// Names are what it is called on PATH, most likely first.
	Names []string `json:"names"`
}

// ModelArg picks the flag that names the model, {model_flag}, by whether
// MODEL is a Hub repo or a file on disk. A model is a file when it is an
// absolute path or ends in one of LocalSuffixes.
type ModelArg struct {
	Hub           string   `json:"hub"`
	Local         string   `json:"local"`
	LocalSuffixes []string `json:"local_suffixes,omitempty"`
}

// Bind is the address the server listens on, {host}. A server that checks a
// key can listen on every interface; one without a key stays on loopback so
// an unset key never leaves the endpoint open.
type Bind struct {
	WithKey    string `json:"with_key"`
	WithoutKey string `json:"without_key"`
}

// Health is how readiness is read.
type Health struct {
	// Path answers 200 when the server is up.
	Path string `json:"path"`
	// Mode is "ready" when Path answers 200 only once the model is loaded,
	// and "live" when it answers as soon as the process listens.
	Mode string `json:"mode"`
}

// Container is how an engine runs in a container on each node of a recipe.
type Container struct {
	// Entrypoint replaces the image's own entrypoint. Empty keeps the
	// image's, and the role's args then start with the command. A role can
	// name its own.
	Entrypoint string `json:"entrypoint,omitempty"`
	// Flags are the docker run flags every rank gets.
	Flags []string `json:"flags,omitempty"`
	// Mounts are host directories, relative to the node user's home,
	// mounted so a relaunch finds weights and compiled kernels warm. A
	// host path may use the engine's params.
	Mounts []Mount `json:"mounts,omitempty"`
	// HFHome is where the image reads the Hugging Face cache, which one of
	// the mounts provides. It is what {weights_path} and the other model
	// paths are built under, and where a launch checks a pinned revision
	// is already downloaded.
	HFHome string `json:"hf_home,omitempty"`
	// ContextFlag is the flag a recipe's args set the context window with,
	// which is what Coder is told the model serves.
	ContextFlag string `json:"context_flag,omitempty"`
	// APIKey is "env:NAME" or "none". Docker is told only the variable's
	// name, so the key is on no command line, docker's included.
	APIKey string `json:"api_key"`
	// LaunchArgs end a recipe's own args, before local overrides are
	// merged: what the launcher owns rather than the model, such as the
	// host and port.
	LaunchArgs []string `json:"launch_args,omitempty"`
	// Roles split the ranks: one role for all of them, or one for rank 0
	// and one for the rest.
	Roles []Role `json:"roles"`
	// LaunchOrder is "workers_first" when the head's rendezvous needs the
	// workers listening, or "head_first".
	LaunchOrder string `json:"launch_order"`
}

// Mount is one host directory mounted into the container.
type Mount struct {
	Host      string `json:"host"`
	Container string `json:"container"`
}

// Role is the command one set of ranks runs.
type Role struct {
	Name string `json:"name"`
	// Ranks is "all", "first" (rank 0) or "rest".
	Ranks string `json:"ranks"`
	// ServesHTTP marks the role whose rank 0 answers the API. Exactly one
	// role has it.
	ServesHTTP bool   `json:"serves_http,omitempty"`
	Entrypoint string `json:"entrypoint,omitempty"`
	// Args is the argv after the entrypoint. A token that is exactly
	// {args} expands to the recipe's merged args.
	Args []string          `json:"args"`
	Env  map[string]string `json:"env,omitempty"`
	// NoAPIKey keeps the key away from ranks that serve nothing.
	NoAPIKey bool `json:"no_api_key,omitempty"`
}

// Capabilities are the differences between engines that shared code acts
// on.
type Capabilities struct {
	// APIKey reports whether the server enforces VLLM_API_KEY itself. It
	// follows from api_key: without one, anything that can reach the port
	// can use the endpoint.
	APIKey bool `json:"-"`
	// VLLMArgs reports whether a profile's settings are vllm serve's. Only
	// then do model-card flags, parser validation against the installed
	// vLLM, KV cache sizing, the memory-utilization fit and the hardware
	// definitions' forbidden flags apply.
	VLLMArgs bool `json:"vllm_args,omitempty"`
	// GGUFQuant reports whether the server loads one GGUF file out of a
	// repo that holds several, named by appending its quant to the model:
	// "org/model-GGUF:Q8_0". Only then does choosing a quant mean anything.
	GGUFQuant bool `json:"gguf_quant,omitempty"`
}

// Placeholders the templates may use, besides an engine's own params.
var (
	processVars   = []string{"bin", "model", "model_flag", "host", "port", "max_model_len", "mem_utilization", "kv_cache_bytes", "api_key"}
	containerVars = []string{"model", "port", "rank", "nodes", "master_addr", "master_port", "api_key"}
	// argVars are the container placeholders a recipe's args and the
	// container's launch_args may use: the ones every rank shares. Those
	// args are filled once, before local overrides are merged in, so a
	// rank's own values are not known yet.
	argVars = []string{"model", "port", "nodes"}
	// modelPathVars are the snapshot paths of a recipe's pinned models,
	// by role, inside the container.
	modelPathVars = []string{"weights_path", "drafter_path", "adapter_path"}
)

// ModelRoles are the roles a recipe's pinned model can play.
var ModelRoles = []string{"weights", "drafter", "adapter"}

// ModelRef is a Hugging Face repo pinned to a revision.
type ModelRef struct {
	ID       string
	Revision string
}

// SnapshotPath is where a pinned revision lives in a Hugging Face cache
// rooted at home: the layout huggingface_hub writes and reads offline.
func SnapshotPath(home string, m ModelRef) string {
	return strings.TrimRight(home, "/") + "/hub/models--" + strings.ReplaceAll(m.ID, "/", "--") + "/snapshots/" + m.Revision
}

// ArgsToken is the role token that expands to the recipe's merged args.
const ArgsToken = "{args}"

var (
	engineIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)
	paramPattern    = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)
	envNamePattern  = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	placeholderRE   = regexp.MustCompile(`\{([a-z][a-z0-9_]*)\}`)
)

// keyMode is a parsed api_key.
type keyMode struct {
	kind string // flag, env or none
	name string // the flag or variable
}

func parseKeyMode(s string) (keyMode, error) {
	if s == "none" {
		return keyMode{kind: "none"}, nil
	}
	kind, name, ok := strings.Cut(s, ":")
	switch {
	case !ok:
	case kind == "flag" && strings.HasPrefix(name, "-"):
		return keyMode{kind: kind, name: name}, nil
	case kind == "env" && envNamePattern.MatchString(name):
		return keyMode{kind: kind, name: name}, nil
	}
	return keyMode{}, fmt.Errorf("api_key %q must be flag:--name, env:NAME or none", s)
}

func (e Engine) keyMode() keyMode {
	m, _ := parseKeyMode(e.APIKey)
	return m
}

// Has reports whether the engine can run in a runtime.
func (e Engine) Has(runtime string) bool { return slices.Contains(e.Runtime, runtime) }

// Name is the PROVIDER value that selects the engine.
func (e Engine) Name() string { return e.ID }

// BinKey is the config key holding the full path to the server executable,
// and "" for an engine with no process runtime.
func (e Engine) BinKey() string {
	if e.Bin == nil {
		return ""
	}
	return e.Bin.Key
}

// BinName is the executable's name on PATH.
func (e Engine) BinName() string {
	if e.Bin == nil || len(e.Bin.Names) == 0 {
		return ""
	}
	return e.Bin.Names[0]
}

// Capabilities reports what shared code has to act on.
func (e Engine) Capabilities() Capabilities {
	c := e.Caps
	if e.Has(RuntimeProcess) {
		c.APIKey = e.keyMode().kind != "none"
	} else {
		c.APIKey = e.ContainerKeyEnv() != ""
	}
	return c
}

// InstallHint says how to get the server on goos.
func (e Engine) InstallHint(goos string) string {
	if h := e.Install[goos]; h != "" {
		return h
	}
	return e.Install["any"]
}

// ParamValues resolves the engine's params: each one's default, replaced by
// set where set names it.
func (e Engine) ParamValues(set map[string]string) map[string]string {
	out := make(map[string]string, len(e.Params))
	for _, p := range e.Params {
		out[p.Name] = p.Default
		if v, ok := set[p.Name]; ok {
			out[p.Name] = v
		}
	}
	return out
}

// Command is the argv that serves s on the host, starting with s.Bin. It
// assumes s passed Validate.
func (e Engine) Command(s Spec) ([]string, error) {
	if !e.Has(RuntimeProcess) {
		return nil, fmt.Errorf("%s runs only in a container; launch it from a recipe", e.ID)
	}
	vals := e.processValues(s)
	args := expandAll(e.Serve, vals)
	for _, group := range e.OptionalArgs {
		if anyEmpty(group, vals) {
			continue
		}
		args = append(args, expandAll(group, vals)...)
	}
	if s.ExtraArgs == "" {
		return args, nil
	}
	words, err := config.SplitArgs(s.ExtraArgs)
	if err != nil {
		return nil, fmt.Errorf("EXTRA_ARGS: %w", err)
	}
	return append(args, words...), nil
}

// Env is the KEY=value entries that serve s on the host, added to the
// server's environment over any inherited entry of the same name. A key
// given as env:NAME is passed here, where ps cannot show it to other users.
func (e Engine) Env(s Spec) []string {
	vals := e.processValues(s)
	var out []string
	if m := e.keyMode(); m.kind == "env" && s.APIKey != "" {
		out = append(out, m.name+"="+s.APIKey)
	}
	for _, k := range slices.Sorted(maps.Keys(e.EnvVars)) {
		if v := expand(e.EnvVars[k], vals); v != "" {
			out = append(out, k+"="+v)
		}
	}
	return out
}

func (e Engine) processValues(s Spec) map[string]string {
	vals := e.ParamValues(s.Params)
	vals["bin"] = s.Bin
	vals["model"] = s.Model
	vals["port"] = s.Port
	vals["max_model_len"] = s.MaxModelLen
	vals["mem_utilization"] = s.MemUtilization
	vals["kv_cache_bytes"] = s.KVCacheBytes
	if e.keyMode().kind == "flag" {
		vals["api_key"] = s.APIKey
	}
	if a := e.ModelArg; a != nil {
		vals["model_flag"] = a.Hub
		if a.isLocal(s.Model) {
			vals["model_flag"] = a.Local
		}
	}
	if b := e.Bind; b != nil {
		vals["host"] = b.WithoutKey
		if s.APIKey != "" {
			vals["host"] = b.WithKey
		}
	}
	return vals
}

// isLocal reports whether model is a file on disk rather than a Hugging Face
// repo: a repo id is never an absolute path.
func (a ModelArg) isLocal(model string) bool {
	if strings.HasPrefix(model, "/") {
		return true
	}
	lower := strings.ToLower(model)
	for _, suf := range a.LocalSuffixes {
		if strings.HasSuffix(lower, suf) {
			return true
		}
	}
	return false
}

// RoleValues are what a container launch knows about one rank.
type RoleValues struct {
	Model      string
	Port       int
	Rank       int
	Nodes      int
	MasterAddr string
	MasterPort int
	// Args are the recipe's merged args, which {args} expands to.
	Args []string
	// Params are the recipe's param settings, over the engine's defaults.
	Params map[string]string
	// Models are the recipe's pinned models by role.
	Models map[string]ModelRef
}

// sharedVars are the values every rank has in common, which is all a
// recipe's args can name.
func (v RoleValues) sharedVars(e Engine) map[string]string {
	vals := e.ParamValues(v.Params)
	if e.Container != nil && e.Container.HFHome != "" {
		for role, m := range v.Models {
			vals[role+"_path"] = SnapshotPath(e.Container.HFHome, m)
		}
	}
	vals["model"] = v.Model
	vals["port"] = itoa(v.Port)
	vals["nodes"] = itoa(v.Nodes)
	return vals
}

func (v RoleValues) vars(e Engine) map[string]string {
	vals := v.sharedVars(e)
	vals["rank"] = itoa(v.Rank)
	vals["master_addr"] = v.MasterAddr
	vals["master_port"] = itoa(v.MasterPort)
	return vals
}

// RoleFor is the role rank runs.
func (e Engine) RoleFor(rank int) (Role, error) {
	if e.Container == nil {
		return Role{}, fmt.Errorf("%s has no container runtime", e.ID)
	}
	for _, r := range e.Container.Roles {
		if r.Ranks == "all" || (r.Ranks == "first") == (rank == 0) {
			return r, nil
		}
	}
	return Role{}, fmt.Errorf("%s has no role for rank %d", e.ID, rank)
}

// LaunchArgs renders the container's launch args. Only v's shared values
// are read.
func (e Engine) LaunchArgs(v RoleValues) []string {
	if e.Container == nil {
		return nil
	}
	return e.ExpandArgs(e.Container.LaunchArgs, v)
}

// ExpandArgs fills a recipe's own args, so a recipe can name a pinned
// model's path, such as --drafter {drafter_path}. Only v's shared values are
// read. Local overrides are merged in after this and never expanded: they are
// free-form, and a literal such as a chat template's {{bos_token}} must reach
// the server as written.
func (e Engine) ExpandArgs(toks []string, v RoleValues) []string {
	return expandAll(toks, v.sharedVars(e))
}

// RenderRole is the entrypoint, argv and environment one rank runs. v.Args
// are placed at {args} as they are, since ExpandArgs already filled the
// recipe's part of them.
func (e Engine) RenderRole(r Role, v RoleValues) (entrypoint string, args []string, env map[string]string) {
	vals := v.vars(e)
	for _, tok := range r.Args {
		if tok == ArgsToken {
			args = append(args, v.Args...)
			continue
		}
		args = append(args, expand(tok, vals))
	}
	env = map[string]string{}
	for k, tmpl := range r.Env {
		if val := expand(tmpl, vals); val != "" {
			env[k] = val
		}
	}
	entrypoint = e.Container.Entrypoint
	if r.Entrypoint != "" {
		entrypoint = r.Entrypoint
	}
	return entrypoint, args, env
}

// RenderMounts resolves the container's mounts for a recipe's params, as
// host paths under home. A param that would take a mount outside home is an
// error, not a mount.
func (e Engine) RenderMounts(home string, params map[string]string) ([]Mount, error) {
	if e.Container == nil {
		return nil, nil
	}
	vals := e.ParamValues(params)
	out := make([]Mount, 0, len(e.Container.Mounts))
	for _, m := range e.Container.Mounts {
		host := expand(m.Host, vals)
		if host == "" || strings.HasPrefix(host, "/") || slices.Contains(strings.Split(host, "/"), "..") {
			return nil, fmt.Errorf("%s: mount %q resolves to %q, which is not a path under the home directory", e.ID, m.Host, host)
		}
		out = append(out, Mount{Host: strings.TrimRight(home, "/") + "/" + host, Container: m.Container})
	}
	return out, nil
}

// HFCacheHostPath is the host directory, relative to home, mounted at the
// container's HFHome, or "" when no mount provides it.
func (e Engine) HFCacheHostPath(params map[string]string) string {
	if e.Container == nil || e.Container.HFHome == "" {
		return ""
	}
	vals := e.ParamValues(params)
	for _, m := range e.Container.Mounts {
		if m.Container == e.Container.HFHome {
			return expand(m.Host, vals)
		}
	}
	return ""
}

// UsesModelPaths reports which model roles the container's templates name,
// so a recipe that does not pin one can be refused before a launch.
func (e Engine) UsesModelPaths() []string {
	if e.Container == nil {
		return nil
	}
	var toks []string
	for _, r := range e.Container.Roles {
		toks = append(toks, r.Args...)
		for _, v := range r.Env {
			toks = append(toks, v)
		}
	}
	return modelRolesIn(toks)
}

// ModelRolesIn reports the model roles a list of tokens names through a
// path placeholder, such as drafter for {drafter_path}.
func ModelRolesIn(toks []string) []string { return modelRolesIn(toks) }

func modelRolesIn(toks []string) []string {
	var out []string
	for _, t := range toks {
		for _, m := range placeholderRE.FindAllStringSubmatch(t, -1) {
			if role, ok := strings.CutSuffix(m[1], "_path"); ok && slices.Contains(ModelRoles, role) && !slices.Contains(out, role) {
				out = append(out, role)
			}
		}
	}
	slices.Sort(out)
	return out
}

// UnknownPlaceholders reports the placeholders a recipe's args name that
// ExpandArgs could not fill for this engine.
func (e Engine) UnknownPlaceholders(toks []string) []string {
	ok := map[string]bool{}
	for _, p := range e.Params {
		ok[p.Name] = true
	}
	for _, v := range argVars {
		ok[v] = true
	}
	if e.Container != nil && e.Container.HFHome != "" {
		for _, v := range modelPathVars {
			ok[v] = true
		}
	}
	var out []string
	for _, t := range toks {
		for _, m := range placeholderRE.FindAllStringSubmatch(t, -1) {
			if !ok[m[1]] && !slices.Contains(out, m[1]) {
				out = append(out, m[1])
			}
		}
	}
	return out
}

// ContainerKeyEnv is the variable a container rank reads the key from, or ""
// when the engine checks none in a container.
func (e Engine) ContainerKeyEnv() string {
	if e.Container == nil {
		return ""
	}
	m, _ := parseKeyMode(e.Container.APIKey)
	if m.kind != "env" {
		return ""
	}
	return m.name
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

func expand(tok string, vals map[string]string) string {
	return placeholderRE.ReplaceAllStringFunc(tok, func(m string) string { return vals[m[1:len(m)-1]] })
}

func expandAll(toks []string, vals map[string]string) []string {
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		out = append(out, expand(t, vals))
	}
	return out
}

func anyEmpty(toks []string, vals map[string]string) bool {
	for _, t := range toks {
		for _, m := range placeholderRE.FindAllStringSubmatch(t, -1) {
			if vals[m[1]] == "" {
				return true
			}
		}
	}
	return false
}

// Registry is a validated set of engine definitions.
type Registry struct {
	engines []Engine
}

// file is the on-disk shape of engines.json and of an engines.d file.
type file struct {
	SchemaVersion int      `json:"schema_version"`
	Engines       []Engine `json:"engines"`
}

func decode(data []byte) (file, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var f file
	if err := dec.Decode(&f); err != nil {
		return file{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return file{}, errors.New("unexpected data after the top-level object")
	}
	if f.SchemaVersion != SchemaVersion {
		return file{}, fmt.Errorf("schema_version is %d; this llm-helper reads version %d", f.SchemaVersion, SchemaVersion)
	}
	return f, nil
}

// Parse reads and validates an engine definitions file.
func Parse(data []byte) (*Registry, error) {
	f, err := decode(data)
	if err != nil {
		return nil, fmt.Errorf("parse engine definitions: %w", err)
	}
	r, err := build(f.Engines, hardware.Shipped())
	if err != nil {
		return nil, fmt.Errorf("invalid engine definitions: %w", err)
	}
	return r, nil
}

// Extend returns a registry with a local file's engines added after this
// one's. A local file can only add engines with new ids: it can teach
// llm-helper a new server, but cannot change how a known one is launched.
func (r *Registry) Extend(data []byte, source string) (*Registry, error) {
	local, err := decode(data)
	if err != nil {
		return nil, err
	}
	merged := slices.Clone(r.engines)
	var errs []error
	for _, e := range local.Engines {
		if _, ok := r.Get(e.ID); ok {
			errs = append(errs, fmt.Errorf("engine %q is already defined; a local file can only add new ids", e.ID))
			continue
		}
		e.Source = source
		merged = append(merged, e)
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return build(merged, hardware.Default())
}

func build(engines []Engine, hw *hardware.Registry) (*Registry, error) {
	var errs []error
	seen := map[string]bool{}
	for _, e := range engines {
		if seen[e.ID] {
			errs = append(errs, fmt.Errorf("engine %q is defined twice", e.ID))
		}
		seen[e.ID] = true
		for _, err := range validate(e, hw) {
			errs = append(errs, fmt.Errorf("engine %q: %w", e.ID, err))
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return &Registry{engines: engines}, nil
}

// validate checks one engine. Every template is checked against the
// placeholders its runtime can fill, so a typo fails here instead of
// reaching the server as an empty argument.
func validate(e Engine, hw *hardware.Registry) []error {
	var errs []error
	add := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }

	if !engineIDPattern.MatchString(e.ID) {
		add("id must be lowercase letters, digits, underscores and dashes")
	}
	if e.DisplayName == "" {
		add("no display_name")
	}
	if strings.TrimSpace(e.Why) == "" {
		add("no why")
	}
	if len(e.Runtime) == 0 {
		add("no runtime")
	}
	for _, rt := range e.Runtime {
		if rt != RuntimeProcess && rt != RuntimeContainer {
			add("runtime %q must be process or container", rt)
		}
	}
	for _, v := range e.GPUVendors {
		if _, ok := hw.Vendor(v); !ok && v != "cpu" {
			add("gpu_vendors names %q, which is not a hardware vendor id or cpu", v)
		}
	}
	if e.Health.Mode != "ready" && e.Health.Mode != "live" {
		add("health.mode %q must be ready or live", e.Health.Mode)
	}
	if !strings.HasPrefix(e.Health.Path, "/") {
		add("health.path %q must start with /", e.Health.Path)
	}

	params := map[string]bool{}
	for _, p := range e.Params {
		switch {
		case !paramPattern.MatchString(p.Name):
			add("param %q must be lowercase letters, digits and underscores", p.Name)
		case slices.Contains(processVars, p.Name) || slices.Contains(containerVars, p.Name) || p.Name == "args":
			add("param %q shadows a built-in placeholder", p.Name)
		case params[p.Name]:
			add("param %q is declared twice", p.Name)
		case strings.TrimSpace(p.Why) == "":
			add("param %q has no why", p.Name)
		}
		params[p.Name] = true
	}
	known := func(base []string, key keyMode, extra ...string) map[string]bool {
		out := maps.Clone(params)
		for _, v := range append(base, extra...) {
			if v == "api_key" && key.kind != "flag" {
				continue
			}
			out[v] = true
		}
		return out
	}
	check := func(where string, toks []string, ok map[string]bool) {
		for _, t := range toks {
			for _, m := range placeholderRE.FindAllStringSubmatch(t, -1) {
				if ok[m[1]] {
					continue
				}
				if m[1] == "api_key" {
					add("%s uses {api_key}, but api_key is not a flag: the key would land on argv, where ps shows it to every user", where)
					continue
				}
				add("%s uses {%s}, which is not a placeholder here", where, m[1])
			}
		}
	}

	var key keyMode
	if e.Has(RuntimeProcess) {
		var err error
		if key, err = parseKeyMode(e.APIKey); err != nil {
			add("%v", err)
		}
	} else if e.APIKey != "" {
		add("api_key belongs to the process runtime; a container's key is container.api_key")
	}
	if e.Has(RuntimeProcess) {
		var avail []string
		if e.ModelArg != nil {
			avail = append(avail, "model_flag")
			if e.ModelArg.Hub == "" || e.ModelArg.Local == "" {
				add("model_arg needs hub and local")
			}
		}
		if e.Bind != nil {
			avail = append(avail, "host")
			if e.Bind.WithKey == "" || e.Bind.WithoutKey == "" {
				add("bind needs with_key and without_key")
			}
			if key.kind == "none" && e.Bind.WithoutKey != e.Bind.WithKey {
				add("bind: a server that checks no key never has one, so with_key would never apply")
			}
		}
		base := slices.DeleteFunc(slices.Clone(processVars), func(v string) bool { return v == "model_flag" || v == "host" })
		ok := known(base, key, avail...)
		switch {
		case e.Bin == nil || e.Bin.Key == "" || len(e.Bin.Names) == 0:
			add("the process runtime needs bin.key and bin.names")
		case !envNamePattern.MatchString(e.Bin.Key):
			add("bin.key %q is not a config key", e.Bin.Key)
		}
		if len(e.Serve) == 0 || e.Serve[0] != "{bin}" {
			add("serve must start with {bin}")
		}
		check("serve", e.Serve, ok)
		for i, g := range e.OptionalArgs {
			if len(g) == 0 {
				add("optional_args[%d] is empty", i)
			}
			check(fmt.Sprintf("optional_args[%d]", i), g, ok)
		}
		checkEnv("env", e.EnvVars, ok, check, add)
		if len(e.Install) == 0 {
			add("the process runtime needs an install hint")
		}
	} else if e.Bin != nil || len(e.Serve) > 0 || len(e.OptionalArgs) > 0 || len(e.EnvVars) > 0 || e.ModelArg != nil || e.Bind != nil {
		add("bin, serve, optional_args, env, model_arg and bind belong to the process runtime, which this engine does not declare")
	}

	if e.Has(RuntimeContainer) {
		ok := known(containerVars, keyMode{kind: "none"})
		argOK := known(argVars, keyMode{kind: "none"})
		if e.Container != nil && e.Container.HFHome != "" {
			for _, v := range modelPathVars {
				ok[v] = true
				argOK[v] = true
			}
		}
		validateContainer(e.Container, ok, argOK, maps.Clone(params), check, add)
	} else if e.Container != nil {
		add("container is set, but runtime does not include container")
	}
	return errs
}

func checkEnv(where string, env map[string]string, ok map[string]bool, check func(string, []string, map[string]bool), add func(string, ...any)) {
	for _, k := range slices.Sorted(maps.Keys(env)) {
		if !envNamePattern.MatchString(k) {
			add("%s: %q is not a variable name", where, k)
		}
		check(where+"."+k, []string{env[k]}, ok)
	}
}

func validateContainer(c *Container, ok, argOK, params map[string]bool, check func(string, []string, map[string]bool), add func(string, ...any)) {
	if c == nil {
		add("the container runtime needs a container section")
		return
	}
	if m, err := parseKeyMode(c.APIKey); err != nil {
		add("container.%v", err)
	} else if m.kind == "flag" {
		add("container.api_key must be env:NAME or none: a flag puts the key on docker's command line")
	}
	if c.LaunchOrder != "workers_first" && c.LaunchOrder != "head_first" {
		add("container.launch_order %q must be workers_first or head_first", c.LaunchOrder)
	}
	hfMounted := false
	for _, m := range c.Mounts {
		if m.Host == "" || strings.HasPrefix(m.Host, "/") || strings.Contains(m.Host, "..") || !strings.HasPrefix(m.Container, "/") {
			add("container.mounts: %q:%q must be a path under the home directory and an absolute container path", m.Host, m.Container)
		}
		check("container.mounts", []string{m.Host}, params)
		hfMounted = hfMounted || (c.HFHome != "" && m.Container == c.HFHome)
	}
	if c.HFHome != "" && !hfMounted {
		add("container.hf_home %q is not a mount's container path, so the pinned models would not be there", c.HFHome)
	}
	if c.ContextFlag != "" && !strings.HasPrefix(c.ContextFlag, "-") {
		add("container.context_flag %q is not a flag", c.ContextFlag)
	}
	// Launch args are filled with the recipe's args, before any rank is
	// known, so they get only the placeholders every rank shares.
	check("container.launch_args", c.LaunchArgs, argOK)
	var all, first, rest, serving int
	for _, r := range c.Roles {
		where := "role " + r.Name
		switch r.Ranks {
		case "all":
			all++
		case "first":
			first++
		case "rest":
			rest++
		default:
			add("%s: ranks %q must be all, first or rest", where, r.Ranks)
		}
		if r.Name == "" {
			add("a role has no name")
		}
		if r.ServesHTTP {
			serving++
			if r.Ranks == "rest" {
				add("%s serves HTTP but does not include rank 0, which the API is reached on", where)
			}
		}
		if len(r.Args) == 0 {
			add("%s has no args", where)
		}
		var toks []string
		for _, t := range r.Args {
			if t != ArgsToken {
				toks = append(toks, t)
			}
		}
		check(where+".args", toks, ok)
		checkEnv(where+".env", r.Env, ok, check, add)
	}
	if !(all == 1 && first == 0 && rest == 0) && !(all == 0 && first == 1 && rest == 1) {
		add("container.roles must be one role for all ranks, or one for the first and one for the rest")
	}
	if serving != 1 {
		add("exactly one role must serve HTTP, not %d", serving)
	}
}

// shippedRegistry is the set compiled into the binary. A parse failure is a
// build defect, caught by every test in every package that imports this one.
var shippedRegistry = func() *Registry {
	r, err := Parse(enginesJSON)
	if err != nil {
		panic(err)
	}
	return r
}()

// Shipped returns the engines compiled into the binary.
func Shipped() *Registry { return shippedRegistry }

// Engines returns every engine, sorted by id.
func (r *Registry) Engines() []Engine {
	out := slices.Clone(r.engines)
	slices.SortFunc(out, func(a, b Engine) int { return strings.Compare(a.ID, b.ID) })
	return out
}

// Get looks an engine up by id.
func (r *Registry) Get(id string) (Engine, bool) {
	for _, e := range r.engines {
		if e.ID == id {
			return e, true
		}
	}
	return Engine{}, false
}
