package cli

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/vllm-helper/internal/config"
	"github.com/coder/vllm-helper/internal/update"
)

// updateChannelKey selects where upgrades come from. It lives in config.env
// so the choice survives the upgrade that acts on it.
const updateChannelKey = "UPDATE_CHANNEL"

func newUpgradeCmd() *cobra.Command {
	var (
		channel string
		check   bool
	)
	c := &cobra.Command{
		Use:     "upgrade",
		Short:   "Update llm-helper itself from a published release",
		GroupID: groupMachine,
		Long: "Replace this binary with the newest build on its channel.\n\n" +
			"stable is the tagged releases. dev is rebuilt from main on every commit,\n" +
			"so it carries unreleased work -- useful for trying a fix before it ships.\n" +
			"A channel given here is remembered for next time.",
		Example: "  vllm-helper upgrade --check\n" +
			"  vllm-helper upgrade\n" +
			"  vllm-helper upgrade --channel dev",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error { return runUpgrade(channel, check) },
	}
	c.Flags().StringVar(&channel, "channel", "", "release channel: stable or dev (default: remembered, else stable)")
	c.Flags().BoolVar(&check, "check", false, "report what's available without installing it")
	return c
}

func runUpgrade(channelFlag string, checkOnly bool) error {
	// Upgrading the binary is not the same as configuring the machine, so
	// this has to work before 'setup' has ever run -- that's exactly when
	// someone might be replacing a binary they just downloaded.
	cfg, err := loadConfig()
	if errors.Is(err, config.ErrNotConfigured) {
		cfg = nil
	} else if err != nil {
		return err
	}

	// A check reports; it doesn't decide anything, so it must not leave the
	// channel changed behind it.
	ch, switched, err := resolveChannel(cfg, channelFlag, !checkOnly)
	if err != nil {
		return err
	}

	token := githubToken(cfg)
	rel, err := update.Latest(ch, token)
	if err != nil {
		return err
	}

	target := releaseVersion(rel, ch)
	fmt.Printf("channel:  %s\n", ch)
	fmt.Printf("running:  %s\n", describeBuild())
	fmt.Printf("available: %s\n", target)

	current := isCurrent(rel, ch)

	// A check answers one question -- is this build what the channel points
	// at -- and that answer doesn't depend on whether --channel happened to
	// differ from the stored preference. Reporting a difference between two
	// identical commits, which is what it did whenever --channel was passed,
	// is simply wrong.
	if checkOnly {
		fmt.Println()
		if current {
			fmt.Println("Already up to date.")
			return nil
		}
		// Deliberately not "an update is available". A dev build is ahead
		// of the newest stable release, so on that channel this is a move
		// back to what's released -- calling it an update would mislead in
		// exactly the case someone is checking carefully.
		fmt.Printf("This differs from what's running. Install it with 'vllm-helper upgrade%s'.\n",
			channelArg(ch, bareChannel(cfg)))
		return nil
	}

	// Installing is a different question. Changing channel is a move to a
	// different build even when the commit happens to match, and it is also
	// what records the choice, so a switch always goes through.
	if !switched && current {
		fmt.Println()
		fmt.Println("Already up to date.")
		return nil
	}

	self, err := os.Executable()
	if err != nil {
		return fmt.Errorf("finding this binary: %w", err)
	}

	fmt.Println()
	fmt.Printf("Downloading %s for %s/%s...\n", target, runtime.GOOS, runtime.GOARCH)
	binary, err := update.Download(rel, runtime.GOOS, runtime.GOARCH, token)
	if err != nil {
		return err
	}
	if err := update.Replace(self, binary); err != nil {
		if errors.Is(err, os.ErrPermission) {
			return fmt.Errorf("%w\n\n%s is not writable by this user -- re-run with sudo:\n  sudo vllm-helper upgrade%s",
				err, self, channelArg(ch, bareChannel(cfg)))
		}
		return err
	}

	// Record what's installed now. Config and profiles outlive the binary,
	// so when a release does need to change something on disk, this is what
	// tells it where the user is coming from.
	previous := recordInstalledVersion(cfg, target)

	fmt.Printf("Installed %s to %s\n", target, self)
	switch {
	case switched:
		fmt.Printf("Channel set to %s in %s\n", ch, cfg.Dirs.ConfigFile())
	case cfg == nil && ch != update.Stable:
		fmt.Printf("There's no config to remember the %s channel in yet; run 'vllm-helper setup',\n", ch)
		fmt.Println("or pass --channel again next time.")
	}
	if previous != "" && previous != target {
		fmt.Printf("Upgraded from %s\n", previous)
	}
	// The old binary was replaced, not the running server: vLLM was exec'd
	// by the previous one and keeps serving until something restarts it.
	fmt.Println()
	fmt.Println("A running server keeps serving with the old build until you restart it:")
	fmt.Println("  vllm-helper restart")
	return nil
}

// githubToken finds a token for reading releases: config and secrets first,
// then the environment, matching how the HuggingFace token is resolved. A
// stored token therefore wins over one in the environment, so a value someone
// deliberately saved isn't quietly displaced by an inherited variable. Empty
// is fine for a public repository.
func githubToken(cfg *config.Config) string {
	if cfg != nil && cfg.GitHubToken() != "" {
		return cfg.GitHubToken()
	}
	for _, k := range []string{"GITHUB_TOKEN", "GH_TOKEN"} {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

// resolveChannel decides which channel to use and reports whether that is a
// change from what was configured. An explicit choice is written to config
// only when persist is set: a read-only command has no business editing it.
func resolveChannel(cfg *config.Config, flag string, persist bool) (update.Channel, bool, error) {
	if cfg == nil {
		// Nothing to read a preference from or write one to; a channel
		// given on the command line still applies to this run.
		if flag == "" {
			return update.Stable, false, nil
		}
		ch, err := update.ParseChannel(flag)
		return ch, false, err
	}
	current, err := update.ParseChannel(cfg.UpdateChannel())
	if err != nil {
		// A bad value in config shouldn't wedge the command when the flag
		// is about to override it anyway.
		if flag == "" {
			return "", false, fmt.Errorf("%s in %s: %w", updateChannelKey, cfg.Dirs.ConfigFile(), err)
		}
		current = ""
	}
	if flag == "" {
		return current, false, nil
	}
	ch, err := update.ParseChannel(flag)
	if err != nil {
		return "", false, err
	}
	if ch == current {
		return ch, false, nil
	}
	// Still report the switch either way: it decides whether an install can
	// be skipped, and whether a suggested command needs to repeat --channel.
	if !persist {
		return ch, true, nil
	}
	cfg.File.Set(updateChannelKey, string(ch))
	if err := cfg.File.Write(cfg.Dirs.ConfigFile(), 0o644); err != nil {
		return "", false, fmt.Errorf("saving %s: %w", updateChannelKey, err)
	}
	return ch, true, nil
}

// isCurrent reports whether the running build is already what the channel
// points at.
//
// The two channels are identified differently on purpose. A stable release
// has a version of its own, so the tag answers it. The dev release reuses one
// moving tag, and its version string sorts below the release it follows, so
// the commit is the only thing that distinguishes two dev builds.
func isCurrent(rel *update.Release, ch update.Channel) bool {
	if ch == update.Dev {
		commit := rel.Commit()
		return commit != "" && Commit != "" && strings.HasPrefix(commit, Commit)
	}
	return strings.TrimPrefix(rel.TagName, "v") == strings.TrimPrefix(Version, "v")
}

func releaseVersion(rel *update.Release, ch update.Channel) string {
	if ch == update.Dev {
		if commit := rel.Commit(); commit != "" {
			return fmt.Sprintf("dev (%s)", shortCommit(commit))
		}
		return "dev"
	}
	return rel.TagName
}

// channelArg repeats --channel in a suggested command when a bare 'upgrade'
// would resolve to something else, so following the suggestion can't quietly
// act on a different channel than the one just reported on.
//
// The test is what a plain run would do now, not whether --channel differed
// from config. Those come apart in two ways: a check deliberately doesn't
// persist, so the stored preference is still the old one; and with no config
// at all there is nowhere to persist to, so every run falls back to stable no
// matter what was asked for.
func channelArg(ch, bare update.Channel) string {
	if ch == bare {
		return ""
	}
	return " --channel " + string(ch)
}

// bareChannel is the channel a plain 'vllm-helper upgrade' would pick right
// now. Call it after resolveChannel: a persisted choice is already reflected
// in cfg, so an install that recorded the channel needs no flag repeated.
func bareChannel(cfg *config.Config) update.Channel {
	if cfg == nil {
		return update.Stable
	}
	ch, err := update.ParseChannel(cfg.UpdateChannel())
	if err != nil {
		return update.Stable
	}
	return ch
}

// describeBuild names the running binary the way its channel identifies it.
func describeBuild() string {
	if Commit == "" {
		return Version
	}
	return fmt.Sprintf("%s (%s)", Version, shortCommit(Commit))
}

func shortCommit(c string) string {
	if len(c) > 12 {
		return c[:12]
	}
	return c
}

// recordInstalledVersion writes the version now on disk and returns whatever
// was recorded before ("" when nothing was).
func recordInstalledVersion(cfg *config.Config, version string) string {
	// The state directory is a property of the machine, not of the config
	// file, so this still works before setup has written one.
	dirs := config.ResolveDirs()
	if cfg != nil {
		dirs = cfg.Dirs
	}
	path := dirs.InstalledVersionFile()
	previous := ""
	if b, err := os.ReadFile(path); err == nil {
		previous = strings.TrimSpace(string(b))
	}
	if err := os.MkdirAll(dirs.State, 0o755); err != nil {
		return previous
	}
	// Best effort: failing to record it must not make a good upgrade look
	// like a failed one.
	_ = os.WriteFile(path, []byte(version+"\n"), 0o644)
	return previous
}
