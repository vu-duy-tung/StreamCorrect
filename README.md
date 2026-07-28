<div align="center">
  <h1>StreamCorrect</h1>
  <p><em>Bringing Offline ASR Performance to Streaming via Error Correction</em></p>

  [![GitHub](https://img.shields.io/badge/GitHub-StreamCorrect-181717?logo=github)](https://github.com/vu-duy-tung/StreamCorrect) [![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org) [![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

  <img src="assets/streamcorrect_overview.png" alt="StreamCorrect overview" width="500" />
</div>

---

## 📖 About

**StreamCorrect** addresses the core challenges of streaming ASR, where error propagation and limited context often degrade performance compared to offline models. It introduces a lightweight error corrector fine-tuned on self-generated data to mitigate accumulated errors in real-time.

Key features:
- 🔁 **Plug-and-play**: Works on top of any offline-based streaming ASR
- ⚡ **Lightweight**: Minimal overhead with a LoRA-fine-tuned correction model
- 🎯 **Effective**: Bridges the gap between offline ASR quality and streaming requirements without distillation

---

## 🎬 Demo

<video src="https://github.com/user-attachments/assets/3a4a2947-5881-4b93-b7b6-f233ee44523b" width="500"></video>

👉 [See more demos](DEMOS.md)

---

## 🚀 Getting Started

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or Miniconda
- CUDA-compatible GPU (recommended)

### Installation

Run the setup script to create the environment, install dependencies, download model checkpoints, and extract datasets:

```bash
bash setup.sh
conda activate StreamCorrect
```

---

## 🧪 Inference

### Single file

Transcribe a single `.wav` file with error correction:

```bash
bash runs/run_single_eval_aishell.sh
```

Override defaults via environment variables:
```bash
AUDIO_PATH=your_audio.wav \
MODEL_PATH=large-v2.pt \
USE_ERROR_CORRECTOR=true \
bash runs/run_single_eval_aishell.sh
```

To disable the error corrector:
```bash
USE_ERROR_CORRECTOR=false bash runs/run_single_eval_aishell.sh
```

### Batch inference

Transcribe a folder of `.wav` files (supports multi-GPU parallel processing):

```bash
AUDIO_DIR=path/to/wavs bash runs/run_batch_eval_aishell.sh
```

### Output format

Results are saved to `save_dir/<run_name>/evaluation_results.json`:

```json
{
  "total_files": 14120,
  "matched_files": 100,
  "average_cer": 0.2171,
  "average_mer": 0.2413,
  "average_first_token_latency_ms": 1731.26,
  "per_file_results": [
    {
      "file": "BAC009S0764W0124.wav",
      "reference": "美国都已经系另外一件事呃欧洲国家亦都系另外一个回事",
      "generated": "都已经 系另外一件事 欧洲国家 亦都系另外一护",
      "cer": 0.24,
      "mer": 0.26,
      "first_token_latency_ms": 1312.21
    }
  ]
}
```

---

## 🏋️ Training

Fine-tune the error corrector on your own data:

```bash
bash SpeechLMCorrector/train.sh --gpus 4
```

---

## 📄 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{streamcorrect2026,
  title   = {StreamCorrect: Bringing Offline ASR Performance to Streaming via Error Correction},
  author  = {},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://arxiv.org}
}
```

---

## 🙏 Acknowledgement

This codebase is built upon the [SimulStreaming](https://aclanthology.org/2025.iwslt-1.41/) project. We thank the authors for their excellent work:

```bibtex
@inproceedings{simulstreaming,
  title     = {Simultaneous Translation with Offline Speech and {LLM} Models in {CUNI} Submission to {IWSLT} 2025},
  author    = {Mach{\'a}{\v{c}}ek, Dominik and Pol{\'a}k, Peter},
  booktitle = {Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)},
  month     = jul,
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.iwslt-1.41/},
  doi       = {10.18653/v1/2025.iwslt-1.41},
  pages     = {389--398}
}
```
