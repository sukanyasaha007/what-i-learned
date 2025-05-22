# 🧠 Mixture of Experts (MoE): Efficient Scaling for Large Models

Mixture of Experts (MoE) is a powerful technique used to **scale transformer models** to trillions of parameters, while keeping computation and latency low. Instead of using all parameters for every input, MoE dynamically routes each token to a **small subset of specialized subnetworks (experts)**.

Used in models like GShard, Switch Transformer, and GLaM, MoE allows huge model capacity with only modest compute budgets.

---

## 🎯 Motivation: Bigger ≠ Slower

Transformer performance scales with model size, but the cost (in FLOPs and memory) becomes prohibitive.

MoE answers this challenge by:
- Dividing the model into **expert layers** (each expert is like a small FFN)
- Routing each token to **only a few experts**, based on learned criteria
- Making training sparse, but model capacity massive

💡 Example:
> A 64-expert MoE layer may only activate 2 experts per token → 2/64 compute used per input, but full model capacity benefits during learning.

---

## 📚 Background & Key Papers

| Paper | Contribution |
|-------|--------------|
| [Shazeer et al., 2017 – Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) | Introduced MoE in LSTM/Transformer |
| [GShard (2020)](https://arxiv.org/abs/2006.16668) | Scalable MoE with learned gating |
| [Switch Transformer (2021)](https://arxiv.org/abs/2101.03961) | Single-expert routing for stability |
| [GLaM (2021)](https://arxiv.org/abs/2112.06905) | Combines mixture-of-experts with sparsity and multilingual models |

---

## 🧱 MoE Architecture: How It Works

```text
Input Token
   ↓
+-----------------------------+
| Gating Network              |  → learns which experts should be used
+-----------------------------+
   ↓
 Select Top-k Experts (e.g., top 2)
   ↓
+----------+   +----------+   ← Only these experts run
| Expert 1 |   | Expert 7 |
+----------+   +----------+
   ↓
 Merge outputs (weighted sum)
```
Each expert is an independent feedforward layer (MLP/FFN)

The gating network chooses the top-k experts for each token

This sparsity means we scale model parameters but only use a fraction per input

## 🧮 Under the hood

Mixture of Experts introduces **conditional computation** into neural networks: instead of using all parameters for each input, we activate only a subset (e.g., top-k experts).

---

### 🔢 MoE Layer Formulation

Let:

- `x ∈ ℝᵈ` be the input token representation  
- `E = {E₁, E₂, ..., Eₙ}` be a set of `n` expert networks (usually MLPs)  
- `G(x) ∈ ℝⁿ` be the **gating network** output — a score distribution over experts  
- `k` be the number of experts to activate (top-k)

---


### 📍 1. Gating Function

A simple softmax gating function is commonly used:

```
G(x) = softmax(W_g · x + b_g)
```

Where:
- `W_g ∈ ℝⁿˣᵈ` and `b_g ∈ ℝⁿ` are learnable parameters
- `G(x)` is a soft probability distribution over experts

To enforce **top-k sparsity**, we zero out all but the top-k values in `G(x)` and optionally renormalize them.

---

### 📍 2. Expert Selection and Forward Pass

Let `S(x) ⊂ E` be the `k` experts selected by the top-k gating values.

Each selected expert `Eᵢ` processes the input independently:
```
yᵢ = Eᵢ(x)
```

The final output is a **weighted sum** of expert outputs:

```
y = ∑ᵢ Gᵢ(x) · Eᵢ(x), for i ∈ S(x)
```

This approach allows **conditional execution**, enabling scaling to massive parameter counts while keeping computation sparse.
### 🧮 Load Balancing (Auxiliary Loss)

To prevent "expert collapse" (i.e., only a few experts being used), we add a regularization term that encourages uniform usage:


```
L_aux = (n · ∑ pᵢ² - 1)²
```

Where `pᵢ` is the average routing probability to expert `i` across the batch. This is similar to entropy regularization.

---

### 📉 Sparse vs Dense Comparison

| Model Type   | Parameters Used Per Input | FLOPs Per Token     | Memory Footprint     |
|--------------|----------------------------|----------------------|------------------------|
| Dense Model  | All layers, all weights    | High (all ops active) | High (no skipping)     |
| MoE Model    | Only top-k experts         | ~1/k of full model    | Same total capacity    |

This makes MoE a great candidate for scaling models while maintaining inference and training efficiency.

---

### ⚙️ Parallelism and Scaling Benefits

MoE is **highly parallelizable**:
- Each expert can run on its own GPU
- Expert computations are independent
- Token routing enables distributed load balancing

Common setups include:
- **Expert Parallelism** (1 expert per GPU/node)
- **Token Parallelism** (split tokens across devices)
- **Combined Parallelism** for very large-scale training (e.g., Switch Transformers)

---

### 🧠 Summary

Mixture of Experts enables:
- 🔬 Large model capacity (100B+ parameters)
- ⚡ Efficient token-level computation (top-k routing)
- 🔁 Dynamic specialization of experts
- 🛠️ Modular reasoning & better scaling curves

MoE is a foundation for many frontier models and continues to be refined for better stability, load balancing, and routing strategies.

---

## 🌍 Real-World Applications

| Area                   | Usage                                                     |
|------------------------|------------------------------------------------------------|
| 💬 Large Language Models | Used in GLaM, GPT-MoE, Switch Transformer                   |
| 🧠 Multilingual Models   | Different experts per language or domain                   |
| ⚡ Inference Efficiency  | Activate only parts of the model at test time              |
| 🛠️ LLM-as-a-Service      | Efficient serving with reduced compute per user            |

## 📌 Summary
    MoE = More parameters + Sparse compute

    Makes large models trainable and deployable

    Plays nicely with distributed + parallel training setups

    Increasingly common in trillion-parameter LLMs