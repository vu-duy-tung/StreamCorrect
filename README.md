# StreamCorrect: Bringing Offline ASR Performance to Streaming via Error Correction

#### StreamCorrect
StreamCorrect is an error correction model for streaming ASR that learns to map erroneous partial hypotheses produced by streaming ASR systems to corrected transcripts. It fine-tunes a Speech Language Model to leverage its strong language modeling capability for post-editing high-quality offline ASR outputs in real time. StreamCorrect is integrated with SimulStreaming to refine incremental ASR predictions under strict latency constraints.

#### About SimulStreaming backbone
SimulStreaming implements Whisper model for translation and transcription in
simultaneous mode (which is known as *streaming* in the ASR community).
SimulStreaming uses the state-of-the-art simultaneous policy AlignAtt, which
makes it very fast and efficient.

SimulStreaming merges [Simul-Whisper](https://github.com/backspacetg/simul_whisper/) and [Whisper-Streaming](https://github.com/ufal/whisper_streaming) projects.

SimulStreaming originates as [Charles University (CUNI) submission to the IWSLT
2025 Simultaneous Shared Task](https://arxiv.org/abs/2506.17077). The results show that this system is extremely robust
and high quality. It is among the top performing systems in IWSLT 2025
Simultaneous Shared Task.

## Preparation
### Install packages

```bash
conda create -n streamingasr python=3.10
conda activate streamingasr
bash install.sh
```

### Model checkpoints
Offline ASR models and Error Correction model could be downloaded [here](https://drive.google.com/drive/folders/1h2tOl6gs93SYZo7fTsc1JYmsOyyRZFLf?usp=sharing)

### Data preparation
Download WSYue-ASR-eval for testing:
```bash
git clone https://huggingface.co/datasets/ASLP-lab/WSYue-ASR-eval 
tar -xzf WSYue-ASR-eval/Short/wav.tar.gz
```
Preprocess the data:
```bash
python wsyue_asr_eval.py \
        --input ./WSYue-ASR-eval/Short/content.txt \
        --output ./WSYue-ASR-eval/Short/content.json \
        --audio-dir ./WSYue-ASR-eval/Short/wav_ \
```

### Install model checkpoint
Download `whisper-medium-yue` checkpoint:
```bash
git clone https://huggingface.co/ASLP-lab/WSYue-ASR
mv WSYue-ASR/whisper_medium_yue/whisper_medium_yue.pt ./
rm -rf WSYue-ASR
```

## Inference

- Inference of SimulStreaming with Error Corrector on a single `.wav` file
```bash
bash runs/run_single_eval_aishell.sh
```

- Inference of SimulStreaming with Error Corrector on a folder of `.wav` files
```bash
bash runs/run_batch_eval_aishell.sh
```

- Output file will be saved to `save_dir/streaming_medium-yue_wsyue_results/evaluation_results.json` with format similar to the follows:
```json
{
  "total_files": 14120,
  "matched_files": 100,
  "unmatched_files": 14020,
  "average_cer": 0.2171288845095628,
  "average_mer": 0.2413379123456789,
  "per_file_results": [
    {
      "file": "0000004453.wav",
      "reference": "美国都已经系另外一件事呃欧洲国家亦都系另外一个回事",
      "generated": "都已经 系另外一件事 欧洲国家 亦都系另外一护",
      "cer": 0.24,
      "mer": 0.26,
      "ref_length": 25,
      "gen_length": 23,
      "first_token_latency_ms": 1312.2074604034424
    },
    {
      "file": "0000009941.wav",
      "reference": "咁我哋就改咗个心出嚟啦即系硬呢度啦吓",
      "generated": "咁我 哋就改咗个心 出嚟啦即系 硬呢度啦",
      "cer": 0.05555555555555555,
      "mer": 0.08333333333333333,
      "ref_length": 18,
      "gen_length": 20,
      "first_token_latency_ms": 1300.1341819763184
    }
  ],
  "average_first_token_latency_ms": 1731.2631171236756
}
```

## To-do
- [x] Fix a logic bug of the token buffer   
- [x] Preliminary test SimulStreaming on Mandarin (AIShell-1)
- [x] Preliminary test SimulStreaming on Cantonese (WSYue)
- [x] Add `whisper-medium-yue` 
- [x] Add Last Token Latency
- [x] Add Error Corrector with Ultravox as backbone