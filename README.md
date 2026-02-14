# ⚡️ Benchmarking GPT-5.3-Codex: Attention Optimization

## 🧪 Project Overview
This repository documents a benchmark test of **OpenAI's GPT-5.3-Codex (Spark Model)**.

As a student researcher working with limited compute (NVIDIA T4), I tasked the model with optimizing a standard Multi-Head Attention mechanism to improve inference latency and memory efficiency. The goal was to see if the "Agentic" workflow could identify system-level bottlenecks without human intervention.

## 📊 Benchmark Results (Sequence Length: 2048)
| Implementation | Inference Time | Speedup |
|----------------|----------------|---------|
| Standard (Manual) | 51.61 ms | 1.0x |
| **Codex Optimized** | **33.60 ms** | **1.54x** |

*(Tested on NVIDIA T4 GPU via Google Colab. Results averaged over 50 runs.)*

## ⚠️ The "8k" Limitation (OOM Analysis)
During the initial phase, I attempted to benchmark at `SEQ_LEN = 8192`.
* **Outcome:** `CUDA out of memory` (Required >16GB VRAM).
* **Insight:** While the Codex model correctly generated the code for 8k sequences, the hardware (T4) could not handle the quadratic memory complexity ($O(N^2)$) of the standard attention map.
* **Fix:** I manually steered the agent to reduce `SEQ_LEN` to **2048** to fit within the 15GB effective VRAM limit.

## 🛠️ The Optimization
The Codex model successfully identified that the manual `softmax` calculation in the standard implementation was the primary memory bottleneck.

It replaced the manual forward pass with a **Fused Kernel operation** (leveraging PyTorch's `F.scaled_dot_product_attention`), which utilizes FlashAttention backends where available. This resulted in a **1.54x speedup** and significantly lower peak memory usage.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies (PyTorch, Matplotlib).
3. Run the benchmark script:

```bash
python benchmark.py
