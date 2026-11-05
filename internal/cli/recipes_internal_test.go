package cli

import (
	"bytes"
	"strings"
	"testing"

	"github.com/coder/llm-helper/internal/recipe"
)

// The display name column was a fixed 22 wide, so a longer name such as
// "DeepSeek V4 Flash Vision" shoved its row's layout column out of line.
func TestRecipeListColumnsLineUp(t *testing.T) {
	recipes := []recipe.TestedRecipe{
		{ID: "short", DisplayName: "Short", Nodes: 1},
		{ID: "a-much-longer-recipe-id", DisplayName: "A Display Name Well Past Twenty-Two", Nodes: 2, Verified: recipe.Verification{Hardware: "dgx_spark_gb10 x2"}},
	}
	var buf bytes.Buffer
	none := func(recipe.TestedRecipe) string { return "" }
	if err := writeRecipeRows(&buf, recipes, none); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != len(recipes) {
		t.Fatalf("got %d lines:\n%s", len(lines), buf.String())
	}
	for _, field := range []func(recipe.TestedRecipe) string{
		func(r recipe.TestedRecipe) string { return r.DisplayName },
		verifiedOn,
		func(r recipe.TestedRecipe) string { return r.Summary() },
	} {
		want := strings.Index(lines[0], field(recipes[0]))
		for i, r := range recipes {
			if got := strings.Index(lines[i], field(r)); got != want || got < 0 {
				t.Errorf("%q starts at column %d, want %d:\n%s", field(r), got, want, buf.String())
			}
		}
	}
}
