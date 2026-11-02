package provider

// mlx is mlx-lm's OpenAI-compatible server, mlx_lm.server, which runs on the
// Apple Silicon GPU through MLX.
//
// It reads its tool-call and reasoning formats from the model's tokenizer, so
// a profile needs no parser flags. It has no API key and binds 127.0.0.1 by
// default, so the command leaves the host alone: listening beyond this machine
// is a choice to make in EXTRA_ARGS, knowing the endpoint is then open.
type mlx struct{}

func (mlx) Name() string    { return MLX }
func (mlx) BinKey() string  { return "MLX_LM_BIN" }
func (mlx) BinName() string { return "mlx_lm.server" }

func (mlx) Capabilities() Capabilities { return Capabilities{} }

func (mlx) Command(s Spec) ([]string, error) {
	return appendExtraArgs([]string{s.Bin, "--model", s.Model, "--port", s.Port}, s.ExtraArgs)
}
