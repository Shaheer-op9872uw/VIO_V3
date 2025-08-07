# 🧠 Model Theory - VIO v3

## 🔍 Introduction

VIO v3 is a novel architecture that reimagines sequence modeling beyond attention. While Transformers rely heavily on quadratic self-attention, VIO introduces a *visual-operator-inspired* flow mechanism—designed for faster, deeper, and more efficient token processing.

---

## 🧬 Core Architecture Inspiration

- **🧠 Mamba-style sequence modeling**: Instead of full attention, VIO uses a *selective state-space operator* (SSO) inspired by structured state-space models.
- **🖼️ Visual perception pipeline**: VIO borrows principles from the human eye—like *foveation*, *local focus*, and *temporal prioritization*.
- **🏗️ Composable Modules**: Each VIO block processes inputs through filtered recurrence, causal convolutions, and non-linear activations—forming a hierarchy of perception similar to the visual cortex.

---

## ⚙️ Internal Mechanics

### 🌀 Flow Mechanism (Core VIO Block)

Each input token `x_t` passes through the following pipeline:

1. **Temporal Filter** `F(t)`: Weights recent tokens more than past ones, mimicking short-term focus.
2. **Depthwise Causal Convolution**: Adds positional inductive bias without explicit embeddings.
3. **Selective Gating** `σ(Wx + b)`: Controls which parts of input to process vs skip.
4. **Hidden State Update**: Maintains a running hidden representation across the sequence.
5. **Feedforward Fusion**: Adds non-linear transformations like GELU or SwiGLU.

```math
h_t = σ(W_f * x_t + b_f) ⊙ (Conv1D(x_{<t})) + ResBlock(x_t)
