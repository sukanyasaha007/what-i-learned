## 🔁 RediX Attention & Prefix Trie: Smarter Inference with Shared Prompts

Modern LLM serving suffers from one key inefficiency: **prompt overlap isn’t reused**. If thousands of users start their queries with the same few words, we end up re-computing and caching the same prefix again and again. Even with PagedAttention, this problem remains.

RediX Attention, introduced in the [sgLang](https://arxiv.org/pdf/2312.07104) paper (2024), solves this by organizing prompts into a **prefix tree (trie)**. Prompts that share common prefixes reuse computation for shared parts—just like a compiler deduplicates shared code.

---

### 🌲 Prefix Tree

Let’s say we have these prompts:

- `"Tell me about transformers"`
- `"Tell me about diffusion models"`
- `"Tell me about sgLang"`

Instead of computing KV for each token per prompt, we build a **prefix tree** like this:

```
"Tell"
│
"me"
│
"about"
    ├── "transformers"
    ├── "diffusion"
    └── "sgLang"
```


Now:
- The KV cache for `"Tell me about"` is computed once
- Only `"transformers"`, `"diffusion"`, and `"sgLang"` need separate suffix computation

---

### 💡 RediX Attention: Key Concepts

| Concept             | Role                                                                 |
|---------------------|----------------------------------------------------------------------|
| **Prefix Trie**     | Organizes shared tokens across batch prompts                         |
| **Node KV Cache**   | Stores partial KV for each unique prefix                             |
| **RediX Execution** | Decodes tokens by traversing the trie and caching at each node       |
| **Pointer Mask**    | Maps generated tokens back to their original prompt lineage          |

---

### ⚙️ Implementation Pseudocode

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_cache = None  # KV for this node

def insert(trie, prompt_tokens):
    node = trie
    for tok in prompt_tokens:
        if tok not in node.children:
            node.children[tok] = TrieNode()
        node = node.children[tok]

def decode_with_redix(trie_root):
    # Traverse and decode each node only once
    for token, child in trie_root.children.items():
        if not child.kv_cache:
            child.kv_cache = run_attention(token, parent_kv=trie_root.kv_cache)
        decode_with_redix(child)
```
## 📈 Results from the Paper
Metric	RediX Attention	vLLM (PagedAttention)
Throughput (↑ better)	5x	1x
Memory usage (↓ better)	-30%	Baseline
Latency for long shared prompts	Reduced	High

RediX particularly shines in:

RAG pipelines (same instructions, different contexts)

Chatbots (template-driven queries)

Agentic workflows (shared control flow)

         ┌────────────────────────────────────┐
         │       Traditional Decoding         │
         └────────────────────────────────────┘
```
Prompt 1: [Tell, me, about, transformers]
Prompt 2: [Tell, me, about, diffusion]
Prompt 3: [Tell, me, about, sgLang]
```

- KV computed for all tokens in all 3 prompts, Total: 15 tokens' worth of KV

         ┌────────────────────────────────────┐
         │          RediX Attention           │
         └────────────────────────────────────┘
             ┌── Tell
             │   └── me
             │       └── about
             │           ├── transformers
             │           ├── diffusion
             │           └── sgLang

- KV computed once for 'Tell me about'

- Only compute suffixes separately

- Saves memory + boosts speed


### 🧪 Some Applications of RediX Attention

RediX Attention shines in any setting where many user prompts share a **common structure**, such as:

- 📚 **RAG Pipelines**: Same instruction template with different retrieved context
- 🤖 **Chatbots & Assistants**: Common greeting or instruction followed by user-specific input
- 🔍 **Search + Generation Systems**: “Summarize this document: …” across many documents
- 🧠 **LLM Agents & Tools**: Repeated control instructions with changing parameters
- 🧪 **Few-Shot Learning**: Shared in-context examples with different test prompts
---

#### 🧵 Example: RAG with Shared Prompt Template

Let’s say we’re building a document Q&A system.

Each prompt has this format:

> "Answer the question based on the following context: 
[CONTEXT] Question: [QUESTION]"

If we receive 1,000 user queries simultaneously, and all use the same template, RediX Attention optimizes the computation by reusing the shared prefix:

```"Answer"
└── "the"
└── "question"
└── "based"
└── "on"
└── ...
├── context: A + Q1
├── context: B + Q2
└── context: C + Q3
```


 Instead of calculating KV attention for the full sequence 1,000 times, we compute once for the shared prefix and only diverge at the context and question parts.

---

#### 🧵 Few-Shot Learning with RediX

In few-shot prompting, we often prepend **in-context examples** like this:

```Prompt 1:
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?

Prompt 2:
Q: What is the capital of France?
A: Paris
Q: What is the capital of Spain?

Prompt 3:
Q: What is the capital of France?
A: Paris
Q: What is the capital of Italy?
```


💡 All prompts **share the same in-context example**, and only differ at the last query. RediX constructs a trie like this:

```
Q: What is the capital of France?
└── A: Paris
└── Q: What is the capital of ...
├── Germany
├── Spain
└── Italy
```



#### 🧠 Summary of RediX Benefits

| Use Case               | Benefit                                    |
|------------------------|---------------------------------------------|
| Few-shot learning      | Reuse of shared examples = less recompute  |
| RAG                    | Efficient prefix reuse across retrieved docs |
| Multi-agent control    | Shared scaffolding logic = cleaner KV reuse |
| Search or summarization| Common templated prompts = memory savings  |

RediX effectively exploits the **prefix redundancy** that’s inherent to many structured LLM applications, especially those using prompt engineering or templating.


RediX is especially helpful for **template-driven generation**, which is common in production LLM deployments across industries like:

- Customer support bots
- Financial document analysis
- Legal contract summarization
- Educational tutoring systems
