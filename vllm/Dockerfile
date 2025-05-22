# 1. Pull the image
docker pull vllm/vllm-openai

# 2. Run server
docker run -p 8000:8000 vllm/vllm-openai --model meta-llama/Llama-2-7b-hf

# 3. Test endpoint
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a haiku about transformers.","max_tokens":32}'
