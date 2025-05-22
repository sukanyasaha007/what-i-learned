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