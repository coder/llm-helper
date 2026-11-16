package cli

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/coder/llm-helper/internal/control"
	"github.com/coder/llm-helper/internal/hfcache"
	"github.com/coder/llm-helper/internal/recipe"
)

type importOptions struct {
	all      bool
	dryRun   bool
	cache    string
	recipeID string
	repo     string
	revision string
}

func newImportCmd() *cobra.Command {
	var o importOptions
	c := &cobra.Command{
		Use:   "import <drive-path> [org/model...]",
		Short: "Copy models from an external drive into the Hugging Face cache",
		Long: "Copy models from an external drive, or any folder, into the Hugging Face cache the\n" +
			"server reads, so 'llm-helper new <org/model>' loads them with no download.\n\n" +
			"The drive can hold a Hugging Face cache (also one made by 'llm-helper export'), an\n" +
			"'hf download --local-dir' folder, or a model folder. The last two do not record\n" +
			"which repo they are, so name it with --repo (and --revision unless the folder is\n" +
			"a git clone).\n\n" +
			"Every file is checked against its recorded hash as it is copied, an interrupted\n" +
			"import resumes where it stopped, and nothing already in the cache is overwritten.\n" +
			"With --recipe the models go where that recipe's engine reads them.",
		Example: "  llm-helper import /media/usb\n" +
			"  llm-helper import /media/usb Qwen/Qwen3-8B-FP8\n" +
			"  llm-helper import /media/usb --recipe <id>",
		GroupID: groupModels,
		Args:    cobra.MinimumNArgs(1),
		RunE:    func(_ *cobra.Command, args []string) error { return runImport(args[0], args[1:], o) },
	}
	f := c.Flags()
	f.BoolVar(&o.all, "all", false, "import every complete model on the drive")
	f.BoolVar(&o.dryRun, "dry-run", false, "show what would be copied and stop")
	f.StringVar(&o.cache, "cache", "", "hub cache directory to import into (default: the one the server reads)")
	f.StringVar(&o.recipeID, "recipe", "", "import the models a tested recipe pins, into the cache its engine reads")
	f.StringVar(&o.repo, "repo", "", "org/model of a folder that is not a Hugging Face cache")
	f.StringVar(&o.revision, "revision", "", "commit of a folder that is not a Hugging Face cache or git clone")
	return c
}

func newExportCmd() *cobra.Command {
	var cache string
	var dryRun bool
	c := &cobra.Command{
		Use:   "export <org/model>... <drive-path>",
		Short: "Copy models from the Hugging Face cache onto an external drive",
		Long: "Copy models from the Hugging Face cache the server reads onto an external drive, to\n" +
			"carry to a machine with no internet. The drive gets a plain copy of each file in a\n" +
			"hub/ folder, which any filesystem holds, and 'llm-helper import' reads it back.\n\n" +
			"Only complete models are copied, and each file is checked against its recorded\n" +
			"hash as it is copied. FAT32 cannot hold a file over 4 GiB; format the drive as\n" +
			"exFAT.",
		Example: "  llm-helper export Qwen/Qwen3-8B-FP8 /media/usb",
		GroupID: groupModels,
		Args:    cobra.MinimumNArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			return runExport(args[:len(args)-1], args[len(args)-1], cache, dryRun)
		},
	}
	c.Flags().StringVar(&cache, "cache", "", "hub cache directory to copy from (default: the one the server reads)")
	c.Flags().BoolVar(&dryRun, "dry-run", false, "show what would be copied and stop")
	return c
}

// serverHub is the cache a command works on when --cache is not given.
func serverHub() string {
	if cfg, err := loadConfig(); err == nil {
		return control.HubCache(cfg)
	}
	return fallbackHub()
}

// fallbackHub is the cache of a machine with no config yet.
func fallbackHub() string {
	home, _ := os.UserHomeDir()
	return hfcache.Dir(hfcache.OSEnv, home)
}

// recipeTarget is the models a recipe pins and the hub cache its engine reads.
func recipeTarget(id string) (models []recipe.Model, hub string, err error) {
	tested, err := recipe.LoadTested()
	if err != nil {
		return nil, "", err
	}
	r, ok := tested.Get(id)
	if !ok {
		return nil, "", fmt.Errorf("no tested recipe %q", id)
	}
	cache := r.Engine().HFCacheHostPath(r.Params)
	if cache == "" || len(r.Models) == 0 {
		return nil, "", fmt.Errorf("recipe %s loads no models from a Hugging Face cache", id)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, "", err
	}
	return r.Models, filepath.Join(home, cache, "hub"), nil
}

func runImport(drive string, names []string, o importOptions) error {
	if o.recipeID != "" && (o.cache != "" || o.repo != "") {
		return errors.New("--recipe picks the models and the cache itself, so it cannot be combined with --cache or --repo")
	}
	for _, n := range names {
		if !hfcache.ValidRepo(n) {
			return fmt.Errorf("%q is not an org/model id", n)
		}
	}
	hub := o.cache
	var pinned []recipe.Model
	switch {
	case o.recipeID != "":
		var err error
		if pinned, hub, err = recipeTarget(o.recipeID); err != nil {
			return err
		}
	case hub == "":
		hub = serverHub()
	}

	found, err := hfcache.Scan(drive, hfcache.ScanOptions{Repo: o.repo, Revision: o.revision})
	if err != nil {
		return err
	}
	if len(found.Models) == 0 {
		return fmt.Errorf("found no models at %s", drive)
	}

	var chosen []hfcache.Model
	switch {
	case o.recipeID != "":
		if chosen, err = pickPinned(found.Models, pinned); err != nil {
			return err
		}
	case len(names) > 0:
		if chosen, err = pickNamed(found.Models, names); err != nil {
			return err
		}
	default:
		listModels(found.Models)
		if chosen, err = pickInteractive(found.Models, o.all); err != nil {
			return err
		}
	}
	if len(chosen) == 0 {
		fmt.Println("Nothing selected.")
		return nil
	}

	need := hfcache.Need(chosen, hub)
	fmt.Printf("Importing %d model(s) into %s: %s to copy.\n", len(chosen), hub, hfcache.FormatBytes(need))
	if err := hfcache.CheckSpace(hub, need); err != nil {
		return err
	}
	if o.dryRun {
		for _, m := range chosen {
			fmt.Printf("  would import %s (%s)\n", m.Repo, hfcache.FormatBytes(m.Size()))
		}
		return nil
	}
	return copyModels(chosen, hub, hfcache.Import, "import", func(m hfcache.Model) {
		switch {
		case o.recipeID != "":
			fmt.Printf("%s is staged for recipe %s.\n", m.Repo, o.recipeID)
		default:
			fmt.Printf("Run 'llm-helper new %s' to serve it.\n", m.Repo)
		}
	})
}

func runExport(repos []string, drive, cache string, dryRun bool) error {
	for _, n := range repos {
		if !hfcache.ValidRepo(n) {
			return fmt.Errorf("%q is not an org/model id", n)
		}
	}
	if st, err := os.Stat(drive); err != nil || !st.IsDir() {
		return fmt.Errorf("%s is not a directory: mount the drive first", drive)
	}
	if cache == "" {
		cache = serverHub()
	}
	var models []hfcache.Model
	for _, repo := range repos {
		m, err := hfcache.ScanRepo(cache, repo)
		if err != nil {
			return err
		}
		if !m.Complete() {
			return fmt.Errorf("%s is incomplete in %s, so it would not load on the other machine: %s", repo, cache, strings.Join(m.Problems, "; "))
		}
		models = append(models, m)
	}
	if err := hfcache.CheckDrive(models, drive); err != nil {
		return err
	}
	dest := filepath.Join(drive, "hub")
	fmt.Printf("Exporting %d model(s) to %s: %s to copy.\n", len(models), dest, hfcache.FormatBytes(hfcache.Need(models, dest)))
	if dryRun {
		for _, m := range models {
			fmt.Printf("  would export %s (%s)\n", m.Repo, hfcache.FormatBytes(m.Size()))
		}
		return nil
	}
	return copyModels(models, dest, hfcache.Export, "export", func(m hfcache.Model) {
		fmt.Printf("On the other machine: llm-helper import %s %s\n", drive, m.Repo)
	})
}

// copyModels runs one copy per model and reports each.
func copyModels(models []hfcache.Model, dest string, do func(hfcache.Model, string, hfcache.Progress) (hfcache.Result, error), verb string, done func(hfcache.Model)) error {
	for _, m := range models {
		fmt.Printf("%s %s\n", strings.ToUpper(verb[:1])+verb[1:], m.Repo)
		progress := func(path string, size int64, skipped bool) {
			if !skipped {
				fmt.Printf("  %s (%s)\n", path, hfcache.FormatBytes(size))
			}
		}
		res, err := do(m, dest, progress)
		if err != nil {
			var corrupt *hfcache.CorruptError
			if errors.As(err, &corrupt) {
				return fmt.Errorf("%s %s: %w (the source is damaged; nothing was kept from that file)", verb, m.Repo, err)
			}
			return fmt.Errorf("%s %s: %w (run it again to resume)", verb, m.Repo, err)
		}
		fmt.Printf("  %d file(s) copied, %d already there, %s written\n", res.Copied, res.Skipped, hfcache.FormatBytes(res.Bytes))
		if res.RefKept != "" {
			fmt.Printf("  note: refs/main still names %s, so the copy is not what an offline load of %s reads\n", res.RefKept, m.Repo)
		}
		done(m)
	}
	return nil
}

func listModels(models []hfcache.Model) {
	fmt.Println()
	for i, m := range models {
		state := "complete"
		if !m.Complete() {
			state = "INCOMPLETE: " + strings.Join(m.Problems, "; ")
		}
		fmt.Printf("  %2d) %-52s %9s  %.7s  %s\n", i+1, m.Repo, hfcache.FormatBytes(m.Size()), m.Revision, state)
	}
	fmt.Println()
}

// pickNamed selects the models the user named, which must all be importable.
func pickNamed(found []hfcache.Model, names []string) ([]hfcache.Model, error) {
	var out []hfcache.Model
	for _, n := range names {
		m, ok := findRepo(found, n)
		if !ok {
			return nil, fmt.Errorf("%s is not on the drive", n)
		}
		if !m.Complete() {
			return nil, fmt.Errorf("%s is incomplete on the drive: %s", n, strings.Join(m.Problems, "; "))
		}
		out = append(out, m)
	}
	return out, nil
}

// pickPinned selects the models a recipe pins, each at its pinned revision.
func pickPinned(found []hfcache.Model, pinned []recipe.Model) ([]hfcache.Model, error) {
	var out []hfcache.Model
	for _, p := range pinned {
		m, ok := findRepo(found, p.ID)
		switch {
		case !ok:
			return nil, fmt.Errorf("%s is not on the drive", p.ID)
		case m.Revision != p.Revision:
			return nil, fmt.Errorf("the recipe pins %s at %s, but the drive has %s", p.ID, p.Revision, m.Revision)
		case !m.Complete():
			return nil, fmt.Errorf("%s is incomplete on the drive: %s", p.ID, strings.Join(m.Problems, "; "))
		}
		out = append(out, m)
	}
	return out, nil
}

// pickInteractive selects with --all, or asks, and skips incomplete models.
func pickInteractive(found []hfcache.Model, all bool) ([]hfcache.Model, error) {
	var ok []hfcache.Model
	for _, m := range found {
		if m.Complete() {
			ok = append(ok, m)
		}
	}
	if len(ok) == 0 {
		return nil, errors.New("no complete model on the drive")
	}
	if all {
		return ok, nil
	}
	if !stdinIsTTY() {
		return nil, errors.New("name the models to import, or pass --all")
	}
	answer := ask("Import which? (numbers like 1,3, or 'all'; Enter to cancel)", "")
	if answer == "" {
		return nil, nil
	}
	if answer == "all" {
		return ok, nil
	}
	var out []hfcache.Model
	for _, field := range strings.Split(answer, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || n < 1 || n > len(found) {
			return nil, fmt.Errorf("%q is not a number from the list", field)
		}
		if !found[n-1].Complete() {
			return nil, fmt.Errorf("%s is incomplete on the drive", found[n-1].Repo)
		}
		out = append(out, found[n-1])
	}
	return out, nil
}

func findRepo(models []hfcache.Model, repo string) (hfcache.Model, bool) {
	for _, m := range models {
		if strings.EqualFold(m.Repo, repo) {
			return m, true
		}
	}
	return hfcache.Model{}, false
}
