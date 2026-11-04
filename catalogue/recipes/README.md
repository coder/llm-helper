# Recipes

Each file here is one tested recipe: a launch configuration that was run and measured on named hardware, named after the recipe's id.
A recipe names the engine that runs it, from [`catalogue/engines/`](../engines), in `provider`.

To add a recipe, copy the closest file to `<id>.json`.
Record only what was measured: pin the image by digest and each model by commit, and declare every departure from the upstream it came from in `source.deviations`.
`tools/recipediff` compares each recipe with its upstream.

The fields are described in "Contributing a source" in [docs/recipes.md](../../docs/recipes.md).
A recipe measured on your own hardware can stay local in `~/.config/vllm-helper/recipes.d/`.
