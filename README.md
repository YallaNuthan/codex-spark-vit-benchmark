# ⚡️ Benchmarking GPT-5.3-Codex: Attention Optimization

## 🧪 Project Overview
This repository documents a benchmark test of **OpenAI's GPT-5.3-Codex (Spark Model)**. The goal was to determine if the model could autonomously optimize a standard Multi-Head Attention mechanism for lower latency.

## 📊 Results (Sequence Length: 2048)
| Implementation | Inference Time | Speedup |
|----------------|----------------|---------|
| Standard (Manual) | 52.73 ms | 1.0x |
| **Codex Optimized** | **34.01 ms** | **1.55x** |

*(Tested on NVIDIA T4 GPU via Google Colab)*

## ⚠️ The "8k" Limitation (OOM Analysis)
During the initial phase, I attempted to benchmark at `SEQ_LEN = 8192`.
* **Outcome:** `CUDA out of memory` (Required >16GB VRAM).
* **Insight:** While Codex correctly generated the code for 8k sequences, the hardware (T4) could not handle the quadratic memory complexity ($O(N^2)$) of the attention map.
* **Fix:** Reduced `SEQ_LEN` to 2048 to fit within the 15GB effective VRAM limit.

## 🛠️ The Optimization
The Codex model successfully identified the bottleneck in the manual `softmax` calculation and replaced the forward pass with a fused operation (leveraging `F.scaled_dot_product_attention`), resulting in the 1.55x speedup.

## 🚀 How to Run
```bash
python benchmark.py --seq-len 2048
