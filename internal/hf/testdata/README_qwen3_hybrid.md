For deployment, you can use `sglang>=0.4.6.post1` or `vllm>=0.8.5` or to create an OpenAI-compatible API endpoint:
- SGLang:
    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3
    ```
- vLLM:
    ```shell
    vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
    ```
