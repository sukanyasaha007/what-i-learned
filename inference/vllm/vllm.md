# vLLM Tutorial: Scalable and Efficient LLM Inference with PagedAttention
## Introduction
- This repo is a quickstart guide to [vLLM](https://github.com/vllm-project/vllm), a high-throughput and memory-efficient inference engine for Large Language Models.

- vLLM introduces **PagedAttention**, which allows dynamic batching and memory sharing across sequences, enabling up to **24x** speedup over traditional Hugging Face inference.


- LLMs are powerful—but serving them to multiple users efficiently is hard. Most frameworks use too much memory and aren’t fast enough when handling many parallel requests. That’s where [vLLM](https://github.com/vllm-project/vllm) comes in. It’s a fast and flexible LLM serving library built around a technique called [**PagedAttention**]((https://arxiv.org/abs/2309.06180)). I tested it locally and summarized what I learned here. 

---

## What Problem Does vLLM Solve?

When using standard libraries like Hugging Face Transformers(or any locally hosted LLM) for inference:
- Each user request builds its own memory block (called the KV cache)
- Memory gets fragmented fast (imagine each sentence taking its own seat on the bus)
- You can’t batch requests of different lengths, so your GPU just waits around

This becomes a bottleneck when you want to scale to 100s or 1000s of users.


Before diving deep into PagedAttention let's first jog our memory on a few basic concepts like KV cache and OS paging.

### 🔁 What is Autoregressive Generation?

Most large language models (like GPT-3, LLaMA, etc.) generate text using a method called **autoregressive generation**. This means they generate one token at a time, in sequence, each time using all previously generated tokens as context.

#### 🔧 How It Works:
- You give the model a prompt:  
  `"The cat sat on the"`
- The model predicts the next token, say `"mat"`
- Then it appends `"mat"` to the prompt and predicts the next token
- This continues until a stop condition (like max tokens or EOS)

#### ✍️ Why It’s Called “Autoregressive”:
Because the model **regresses (predicts)** the next value **based on previous outputs**. Mathematically, it tries to model:

P(x₁, x₂, ..., xₙ) = P(x₁) * P(x₂ | x₁) * P(x₃ | x₁, x₂) * ... * P(xₙ | x₁, ..., xₙ₋₁)

So each new token depends on all the tokens before it.


### 🧠 What is KV Cache and Why Do We Need It?

Due to this autoregressive nature of LLM inference, each token attends to all previous tokens to generate the next one. This involves computing attention using the **Key (K)**, **Value (V)**, and **Query (Q)** matrices. During training, this is done in parallel. 

If we recomputed attention from scratch at every step, the cost would grow quadratically with sequence length. That's inefficient.

To solve this, we use a **Key-Value cache**, which stores the K and V matrices for all previously generated tokens. Then, for each new token:
- Only the Query for the new token is computed
- It attends over the cached K/V values from earlier tokens
- This reduces redundant computation and makes inference fast and linear in time

#### ⚡ Example:
- Prompt: "The quick brown fox"
- After token 1: cache K/V for "The"
- After token 2: reuse K/V for "The", cache K/V for "quick"
- …
- The final token only queries against the cached past

Without KV caching, inference would be too slow for real-time apps like chatbots, search completion, or RAG pipelines.  
This is why optimizing KV cache memory usage (like vLLM does) is so critical to scaling LLMs.


### 🧨 What is Memory Fragmentation (in OS and LLMs)?

In an operating system, **memory fragmentation** happens when free memory is broken into small, non-contiguous blocks over time. Even if there's enough total free memory, large allocations may fail because the free chunks are scattered.

🖥️ **In OS memory management:**
- Programs allocate and free memory dynamically.
- Over time, memory becomes fragmented.
- Large programs may not get a big enough block of contiguous memory.

🧠 **In LLM inference (e.g., with Hugging Face Transformers):**
- Each sequence during generation gets its own block of KV memory.
- These KV caches are often **preallocated** and rarely reused efficiently.
- With sequences of different lengths, memory usage becomes uneven.
- You end up wasting a lot of GPU memory that’s allocated but underutilized.

### 🔧 Why it matters:
- Fragmentation limits the number of concurrent sequences you can serve.
- Your GPU gets "full" even when it’s mostly empty.
- vLLM fixes this by using a **paged layout** to dynamically allocate small, reusable blocks.


### 📄 What is PagedAttention?

As your model generates tokens autoregressively, the **KV cache** grows with every step. But here's the problem: traditional frameworks allocate one big contiguous block of memory per sequence—even if that sequence is short or pauses mid-generation. Over time, this leads to **fragmentation** and **wasted GPU memory**.

**PagedAttention** changes that by treating the KV cache like an operating system treats virtual memory. Instead of allocating a large block all at once, it breaks the memory into **fixed-size pages** (like 16 or 32 tokens each). These pages are dynamically assigned and reused across sequences.

#### 🧠 Key Benefits:
- 🧩 **Fine-grained memory control**: Allocate memory at the token level instead of whole sequences.
- 🧠 **Dynamic batching**: Mix sequences of different lengths without padding or waste.
- 🔁 **Better GPU utilization**: Share memory pages across active and paused sequences.

PagedAttention is what makes **vLLM** special—it allows the model to serve **thousands of users concurrently**, without hitting memory limits or sacrificing speed. It’s a simple idea inspired by operating systems, but it solves a very real bottleneck in modern LLM serving.


## 🧩 How it is Like Paging in Operating Systems
If you've taken an Operating Systems course, this will sound familiar:

| 🖥️ **OS Paging**                  | 🧠 **vLLM PagedAttention**                    |
|----------------------------------|----------------------------------------------|
| Memory is divided into fixed-size pages (e.g., 4KB) | KV cache memory is divided into fixed-size pages (token-level granularity) |
| Uses a page table to map virtual memory to physical memory | Uses a page table to map tokens to memory pages |
| Loads only the required pages into RAM (on-demand) | Loads only needed token pages into GPU memory |
| Reduces memory fragmentation and overhead | Reduces key/value cache fragmentation during inference |
| Enables efficient multitasking and dynamic memory reuse | Enables batching sequences of varying lengths dynamically |
| Handles page faults to avoid loading unused memory | Avoids allocating full KV blocks for every prompt |


### 📈 Result:
- Up to **3x less memory usage**
- Up to **24x faster** generation compared to standard approaches

---


## 🧪 Conclusion

The vLLM paper ("vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention") introduces a system optimized for high-throughput, low-latency LLM inference. It emphasizes the importance of **continuous batching**, which allows new user requests to be added dynamically into the same batch without requiring the model to restart or pad for alignment. The heart of vLLM is **PagedAttention**, which enables this by decoupling attention memory from sequence boundaries. The paper demonstrates that vLLM can achieve **up to 24x higher throughput** than Hugging Face Transformers with the same model, and **up to 2.2x better memory efficiency** compared to DeepSpeed’s inference engine. It supports models with tens of billions of parameters, works with popular libraries like Hugging Face and OpenAI-compatible APIs, and integrates seamlessly into serving stacks without requiring model retraining or architecture changes. Overall, the paper presents vLLM as a practical and production-ready solution for scaling LLM inference.


### 📦 Setup (Python)

```bash
pip install vllm
```

### This tutorial is writen by [@sukanyasaha007](linkedin.com/in/sukanyasaha007)(and lots of ☕) and rectified by AI tools