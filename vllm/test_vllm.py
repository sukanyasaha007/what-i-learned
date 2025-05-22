from vllm import LLM, SamplingParams

def run():
    model_name = "meta-llama/Llama-2-7b-hf"
    llm = LLM(model=model_name)

    prompt = ["Explain PagedAttention like I’m five."]
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

    outputs = llm.generate(prompt, sampling_params)
    print("Prompt:", prompt[0])
    print("Output:", outputs[0].outputs[0].text.strip())

if __name__ == "__main__":
    run()