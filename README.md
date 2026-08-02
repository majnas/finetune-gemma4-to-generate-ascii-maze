# Gemma-4 (E2B) ASCII-Maze Finetune

<p align="center">
  <a href="https://majnas.github.io/finetune-gemma4-to-generate-ascii-maze/">
    <img src="assets/varNxN_sample10_topview.gif" alt="varNxN sample 10, top view" />
  </a>
</p>

Finetune `unsloth/gemma-4-E2B-it` on a local ASCII-maze dataset with Unsloth +
LoRA, then export / convert / run the result.

## 🎮 Live 3D maze gallery

**[→ Open the interactive gallery](https://majnas.github.io/finetune-gemma4-to-generate-ascii-maze/)**

An interactive 3D carousel of model-generated ASCII mazes, one tab per
fine-tuning phase (`fixed4x4`, `varNxN`, `varNxM`, `varNxM_rndSE`). Pick a
phase, then drag to orbit / scroll to zoom the centered maze, click a side
preview (or use the arrows / ←→ keys) to browse its 100 samples.

`docs/index.html` is a hand-maintained static page; only the sample data
is generated. Regenerate `docs/data/*.json` from each phase's `gguf_samples*.txt`
after a new batch of samples:

```bash
python3 asciimaze/generate_maze_carousel_data.py
```

Pass `--phase NAME=PATH` (repeatable) to point a phase at a different sample
file. To preview locally, `fetch()` needs a real HTTP origin (opening
`index.html` via `file://` won't load the data), so serve `docs/` instead:

```bash
cd docs && python3 -m http.server 8000
```

`asciimaze/generate_maze_gallery.py` still exists for generating a one-off
full-grid HTML gallery from a single `outputs/*.txt` file, if ever needed:

```bash
python3 asciimaze/generate_maze_gallery.py <path-to-samples.txt> -o maze_gallery.html
```

## Repo layout

```
asciimaze/
├── maze/                     # shared maze generator/solver/renderer
├── gemma4_train.py  gemma4_export.py  gemma4_gguf.py
├── gemma4_inference.py  gemma4_generate_100_samples_*.py
├── fixed4x4/                 # the fixed 4x4 maze experiment
│   ├── build_dataset.py      # generates data/*.jsonl
│   ├── config.yaml           # train/export/gguf config
│   ├── data/                 # train/val/test.jsonl
│   ├── finetune/{lora,merged,gguf}/
│   └── outputs/              # batch-generation results
├── varNxN/                   # variable square-size maze experiment (3x3-9x9), reuses maze/
├── varNxM/                   # variable rectangular maze experiment (rows != columns), reuses maze/
└── varNxM_rndSE/             # same as varNxM but S/E are random cells, not fixed corners
```

Run all commands from the repo root. Path flags default into
`asciimaze/fixed4x4/`; point them at another folder to target a different
experiment.

## Setup

Copy `.env.example` to `.env` and set `HF_CACHE_DIR` (routes the Hugging Face
cache to a writable location instead of a shared/permission-locked one).

## 0. Generate the dataset

```bash
python -m asciimaze.fixed4x4.build_dataset --n 7000 --seed 1934
```

Writes `train`/`val`/`test` JSONL splits (90/5/5) to `asciimaze/fixed4x4/data/`.

## Config file (`asciimaze/fixed4x4/config.yaml`)

`gemma4_train.py`/`gemma4_export.py`/`gemma4_gguf.py` carry no literal
defaults - every value comes from this file's `train:`/`export:`/`gguf:`
sections (auto-loaded via `--config`'s own default), and a shared
`experiment_dir` is templated into each section's paths. Any CLI flag
overrides its config value for that run.

## 1. Train (LoRA)

```bash
python asciimaze/gemma4_train.py --gpu 3
```

Trains with SFTTrainer, saves adapters to `asciimaze/fixed4x4/finetune/lora`.

## 2. Export merged float16 model

```bash
python asciimaze/gemma4_export.py --gpu 3
```

Merges the LoRA adapters into a standalone float16 checkpoint at
`asciimaze/fixed4x4/finetune/merged`.

## 3. Convert to GGUF (llama.cpp)

```bash
python asciimaze/gemma4_gguf.py --gpu 3
```

Produces `asciimaze/fixed4x4/finetune/gguf/gemma-4-E2B-it.F16.gguf` + a
`Modelfile` for Ollama.

> If merged/GGUF output looks like the un-finetuned base model, it's a
> known low-rank LoRA merge-loss issue - see the config's `lora_r`/`lora_alpha`
> (kept at 16/32 to avoid it), or serve the adapter unmerged instead.

## 4. Quick inference (single prompt)

```bash
python asciimaze/gemma4_inference.py --gpu 3
```

## Batch generation - 100 samples

```bash
python asciimaze/gemma4_generate_100_samples_with_lora.py --gpu 3
python asciimaze/gemma4_generate_100_samples_with_merged.py --gpu 3
python asciimaze/gemma4_generate_100_samples_with_gguf.py --gpu 3
```

Each generates N independent samples of the same prompt to one output file
under `asciimaze/fixed4x4/outputs/`.

## Serve the GGUF as an API / chat UI

```bash
CUDA_VISIBLE_DEVICES=3 ~/.unsloth/llama.cpp/llama-server \
  -m asciimaze/fixed4x4/finetune/gguf/gemma-4-E2B-it.F16.gguf \
  -ngl 99 -c 4096 --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090`, or POST to `/v1/chat/completions`
(OpenAI-compatible).
