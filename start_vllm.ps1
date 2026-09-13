.\venv\Scripts\activate
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ --gpu-memory-utilization 0.55 --enable-prefix-caching
