package cli

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/coder/llm-helper/internal/config"
	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/profile"
	"github.com/coder/llm-helper/internal/recipe"
	"github.com/coder/llm-helper/internal/service"
)

// clusterRunner drives a recipe that serves from containers on the nodes it
// was verified on. It satisfies the same contract as the init-system service
// so status, logs, stop and restart do not have to know which one is serving:
// before this, they reported systemd even while a container did the work.
type clusterRunner struct {
	recipe recipe.TestedRecipe
	opts   recipeLaunchOptions
}

// Both runners must stay interchangeable; the compiler is the only reliable
// guard against one of them drifting out of the contract.
var (
	_ runner = (*clusterRunner)(nil)
	_ runner = service.Service{}
)

// clusterRunnerFor builds the runner from what the launch recorded on the
// profile. A profile that predates that recording cannot be driven safely, so
// this says so instead of guessing a node list.
func clusterRunnerFor(cfg *config.Config, p *profile.Profile) (*clusterRunner, error) {
	r, ok := recipeForProfile(p)
	if !ok {
		return nil, fmt.Errorf("profile %q is not owned by a tested recipe", p.Name)
	}
	// A profile written before the topology was recorded still gets a runner:
	// reading it back as the init system is how status and logs came to report
	// a dead unit while a container served. Read-only verbs degrade to "we
	// cannot see it"; the verbs that would act refuse in needNodes.
	o := recipeLaunchOptions{
		nodeNames:  p.Nodes(),
		container:  p.Container(),
		sshKey:     p.SSHKey(),
		masterPort: p.MasterPort(),
	}
	if o.container == "" {
		o.container = defaultRecipeContainer
	}
	if o.masterPort == 0 {
		o.masterPort = defaultRecipeMasterPort
	}
	return &clusterRunner{recipe: r, opts: o}, nil
}

// needNodes guards the verbs that change something. Guessing a node list
// would start or stop containers on machines nobody named.
func (c *clusterRunner) needNodes() error {
	if len(c.opts.nodeNames) == 0 {
		return fmt.Errorf("the active profile does not record the nodes %s was launched on; relaunch it with 'llm-helper recipes launch %s --nodes <n1,n2>'", c.recipe.ID, c.recipe.ID)
	}
	return nil
}

// preflight is everything a launch would refuse that can be checked without
// touching a node: a recorded topology, nodes that are still in the registry
// at the recipe's own count, and resolved args that the recipe's hardware and
// every node's own recorded architecture can run.
// Activation runs it before publishing to Coder, because a refusal after that
// point leaves Coder advertising a model nobody is serving.
func (c *clusterRunner) preflight(cfg *config.Config) error {
	if err := c.needNodes(); err != nil {
		return err
	}
	members, err := resolveRecipeMembers(c.recipe, c.opts)
	if err != nil {
		return err
	}
	local, err := recipeOverrides(cfg, c.recipe)
	if err != nil {
		return err
	}
	args, err := c.recipe.ServeArgsWith(local.args)
	if err != nil {
		return err
	}
	return checkRecipeMembers(os.Stderr, c.recipe, members, args)
}

// endpointURL is where the head serves the API. The head can be another
// machine, so this is the node's address on the recipe's port rather than
// localhost on whatever port config names.
func (c *clusterRunner) endpointURL() (string, error) {
	m, err := c.head()
	if err != nil {
		return "", err
	}
	host := "localhost"
	if !m.node.IsLocal() {
		host = m.node.Address()
	}
	return "http://" + net.JoinHostPort(host, strconv.Itoa(c.recipe.Port)), nil
}

// probeHealth asks the head whether it serves, on the recipe's health path.
// When the head is another machine the probe runs on the head itself: an
// engine that listens only on loopback, as TensorFold does unless told
// otherwise, cannot be reached from anywhere else.
func (c *clusterRunner) probeHealth() (int, bool) {
	m, err := c.head()
	if err != nil {
		return 0, false
	}
	port := strconv.Itoa(c.recipe.Port)
	path := c.recipe.HealthPath()
	if path == "" {
		path = "/health"
	}
	if m.node.IsLocal() {
		return control.ProbeHealthPath("http://localhost:"+port, path)
	}
	out, err := nodeScript(m, c.opts.sshKey, headProbeScript(port, path, "127.0.0.1", m.node.Address()), nil)
	if err != nil {
		return 0, false
	}
	code, _ := strconv.Atoi(strings.TrimSpace(out))
	return code, code == http.StatusOK
}

// headProbeScript prints the HTTP status the first of hosts that answers 200
// gave for path, or the last status seen, or 0. It needs only bash, since a
// node is not promised curl.
func headProbeScript(port, path string, hosts ...string) string {
	var quoted []string
	for _, h := range hosts {
		if h != "" {
			quoted = append(quoted, recipe.ShellQuote(h))
		}
	}
	const get = `exec 3<>"/dev/tcp/$1/$2" || exit 1
printf "GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" "$3" "$1" >&3
read -r _ code _ <&3
echo "${code%[[:space:]]}"`
	return fmt.Sprintf(`t=
command -v timeout >/dev/null && t="timeout 3"
code=0
for h in %s; do
	c=$($t bash -c '%s' _ "$h" %s %s 2>/dev/null)
	[ -n "$c" ] && code=$c
	[ "$code" = 200 ] && break
done
echo "$code"
`, strings.Join(quoted, " "), get, recipe.ShellQuote(port), recipe.ShellQuote(path))
}

// identity names what is holding the GPUs, so a switch can tell whether the
// runner it is leaving is the one it is about to start. Two recipes on the
// same nodes and container name replace each other; anything else does not.
func (c *clusterRunner) identity() string {
	return "cluster:" + c.opts.container + "@" + strings.Join(c.opts.nodeNames, ",")
}

// head is the node serving the API: rank 0.
func (c *clusterRunner) head() (recipeMember, error) {
	members, err := resolveRecipeMembers(c.recipe, c.opts)
	if err != nil {
		return recipeMember{}, err
	}
	return members[0], nil
}

// inspect reads the head container. Absent is not an error: a stopped cluster
// is a normal state that status has to be able to describe.
func (c *clusterRunner) inspect() (status string, exitCode, restarts int) {
	m, err := c.head()
	if err != nil {
		return "", 0, -1
	}
	script := "docker inspect " + recipe.ShellQuote(c.opts.container) +
		" --format '{{.State.Status}} {{.State.ExitCode}} {{.RestartCount}}' 2>/dev/null || true"
	out, err := nodeScript(m, c.opts.sshKey, script, nil)
	if err != nil {
		return "", 0, -1
	}
	fields := strings.Fields(out)
	if len(fields) < 3 {
		return "", 0, -1
	}
	code, _ := strconv.Atoi(fields[1])
	n, err := strconv.Atoi(fields[2])
	if err != nil {
		n = -1
	}
	return fields[0], code, n
}

// IsActive speaks the init system's vocabulary so callers can treat both
// runners alike. "activating" matters most: a model can take minutes to load,
// and reporting that as inactive sends people looking for a fault.
func (c *clusterRunner) IsActive() string {
	status, code, _ := c.inspect()
	return containerState(status, code)
}

// containerState translates docker's vocabulary into the init system's. A
// clean exit is "inactive" rather than "failed" because that is what stopping
// the cluster on purpose looks like, and an empty status means no container at
// all, which is the same thing to a caller.
func containerState(status string, exitCode int) string {
	switch status {
	case "running":
		return "active"
	case "created", "restarting":
		return "activating"
	case "":
		return "inactive"
	case "exited", "dead":
		if exitCode != 0 {
			return "failed"
		}
		return "inactive"
	default:
		return status
	}
}

func (c *clusterRunner) NRestarts() int {
	_, _, n := c.inspect()
	return n
}

func (c *clusterRunner) Health() service.Health {
	status, code, restarts := c.inspect()
	h := service.Health{State: c.IsActive(), SubState: status, NRestarts: restarts}
	if (status == "exited" || status == "dead") && code != 0 {
		h.Result = "exit-code " + strconv.Itoa(code)
		h.Crashed = true
	}
	if status == "restarting" {
		h.Crashed = true
	}
	return h
}

// Restart relaunches every rank. It reuses the launch path with activation
// skipped: the profile and Coder already describe this recipe, and re-syncing
// Coder on every watchdog recovery would write to it for no change.
func (c *clusterRunner) Restart() error {
	if err := c.needNodes(); err != nil {
		return err
	}
	return runRecipeLaunch(context.Background(), c.recipe.ID, c.restartOpts())
}

func (c *clusterRunner) restartOpts() recipeLaunchOptions {
	o := c.opts
	o.skipCoder = true
	o.relaunch = true
	return o
}

func (c *clusterRunner) Stop() error {
	if err := c.needNodes(); err != nil {
		return err
	}
	return runRecipeStop(c.recipe.ID, c.opts)
}

// logScript tails the head container. The engine logs are what explain a slow
// or failed load, and they live in the container, not in the host journal.
func (c *clusterRunner) logScript(n int, follow bool) string {
	s := "docker logs --tail " + strconv.Itoa(n)
	if follow {
		s += " -f"
	}
	return s + " " + recipe.ShellQuote(c.opts.container) + " 2>&1"
}

func (c *clusterRunner) LastLogs(n int) string {
	m, err := c.head()
	if err != nil {
		return ""
	}
	out, _ := nodeScript(m, c.opts.sshKey, c.logScript(n, false), nil)
	return out
}

func (c *clusterRunner) PrintLogs(n int) error {
	fmt.Print(c.LastLogs(n))
	return nil
}

// StartTail starts a follower and hands back the command so the caller can
// kill it, matching the init-system runner. It goes through nodeCommand so a
// head on another machine is tailed over ssh; killing the command closes that
// hop, which ends the remote follow.
func (c *clusterRunner) StartTail(n int) (*exec.Cmd, error) {
	m, err := c.head()
	if err != nil {
		return nil, err
	}
	cmd, err := nodeCommand(m, c.opts.sshKey, c.logScript(n, true), nil)
	if err != nil {
		return nil, err
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// Label says "cluster" rather than the init system, because that is what is
// actually holding the model up.
func (c *clusterRunner) Label() string { return "cluster" }

func (c *clusterRunner) LogHint() string {
	return "docker logs " + c.opts.container + " on the head node"
}
