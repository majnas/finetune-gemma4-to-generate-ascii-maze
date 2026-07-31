# Gemma-4 (E2B) ASCII-Maze Finetune

Finetune `unsloth/gemma-4-E2B-it` on a local ASCII-maze dataset with Unsloth +
LoRA, then export / convert / run the result. Split out of the original
`gemma4_(e2b)_text.py` notebook into standalone, argparse-driven scripts.

## Repo layout

Everything lives under one `asciimaze` package. Maze generation itself is
**shared** across experiments: `asciimaze/maze/` holds the size-agnostic maze
generator, solver, and ASCII renderer (parameterized by rows/columns, no
fixed-size assumptions). Each experiment then gets its own sibling folder
inside `asciimaze/` holding that experiment's dataset-building code, config,
data, and model artifacts — currently `asciimaze/fixed4x4/` (fixed 4×4 mazes);
`asciimaze/varNxN/` (variable-size mazes) is scaffolded for a later phase and
will reuse the same `asciimaze/maze/` engine. The `gemma4_*.py` driver scripts
(train/export/gguf/inference/batch-generation) are also shared across
experiments — they take path flags (`--train-file`, `--output-dir`,
`--lora-path`, etc.) rather than hardcoding one experiment — so they live
alongside the experiment folders in `asciimaze/` too:

```
asciimaze/
├── __init__.py
├── maze/                           # shared maze generator/solver/renderer (size-agnostic)
├── gemma4_train.py  gemma4_export.py  gemma4_gguf.py       # shared driver scripts
├── gemma4_inference.py  gemma4_generate_100_samples_*.py
├── fixed4x4/                       # one experiment (fixed 4×4 mazes, also a Python package)
│   ├── __init__.py
│   ├── build_dataset.py            # generates data/*.jsonl (see step 0)
│   ├── config.py  dataset.py  paths.py  prompts.py
│   ├── config.yaml                 # config for train/export/gguf (see below)
│   ├── data/                       # train.jsonl / val.jsonl / test.jsonl
│   ├── finetune/
│   │   ├── lora/      # LoRA adapters        (gemma4_train.py)
│   │   ├── merged/    # merged float16 model (gemma4_export.py --save-merged)
│   │   └── gguf/      # GGUF + Modelfile     (gemma4_gguf.py  --save-gguf)
│   └── outputs/       # batch-generation results (*.txt)
└── varNxN/                         # (planned) variable-size maze experiment, reuses asciimaze/maze/
README.md
```

**All path defaults now point into `asciimaze/fixed4x4/`** (train/val files, model dirs,
output files), so the commands below only pass what differs from the defaults —
mainly `--gpu`. Run everything from the **repo root** (one level above
`asciimaze/`). Examples use **`--gpu 3`**; change it to a free CUDA index
(`nvidia-smi`).

> **Targeting a different experiment:** point the path flags at another folder,
> e.g. `--train-file myexp/data/train.jsonl --output-dir myexp/.../lora`,
> `--lora-path`, `--merged-dir`, `--gguf-dir`, `--model-path`, `--gguf-path`,
> `--output-file`.

## Hugging Face cache

Every script pins the HF cache to your home dir up front:

```python
os.environ["HF_HOME"]           = ~/.cache/huggingface
os.environ["HF_HUB_CACHE"]      = ~/.cache/huggingface/hub       # models
os.environ["HF_DATASETS_CACHE"] = ~/.cache/huggingface/datasets  # datasets + locks
```

This avoids a shared/permission-locked cache. The base model
(`unsloth/gemma-4-E2B-it`, ~10 GB) downloads once into `~/.cache/huggingface/hub`
and is reused by every script afterwards — no re-download.

## 0. Generate the dataset

`asciimaze/maze/` holds the shared, size-agnostic maze generator, solver, and
ASCII renderer (reused by every experiment). `asciimaze/fixed4x4/` is the
fixed-4×4 experiment package (`asciimaze/fixed4x4/__init__.py`) built on top
of it, and its `build_dataset.py` writes the `train`/`val`/`test` JSONL splits
consumed by `gemma4_train.py`. Run it as a module from the repo root:

```bash
python -m asciimaze.fixed4x4.build_dataset --n 7000 --seed 1934
```

Splits `--n` samples 90/5/5 into train/val/test (e.g. `7000` → `6300/350/350`),
each maze generated from `seed + index` for reproducibility, and writes to
`--out` (default `asciimaze/fixed4x4/data/`). Every record has a `conversations` field
(`[{"role": "user", ...}, {"role": "assistant", ...}]`) — the field name
`gemma4_train.py`'s data prep (Unsloth's `standardize_data_formats`) requires;
it silently no-ops on any other field name (e.g. `messages`).

> Maze layout (fixed 4×4 grid, fixed start/end corners, no solution path in
> the output) is controlled by `asciimaze/fixed4x4/config.py`'s `MAZE_CONFIG`, not the
> YAML config — edit it directly to change grid size or randomize endpoints.

## Config file (`asciimaze/fixed4x4/config.yaml`)

`gemma4_train.py`, `gemma4_export.py`, and `gemma4_gguf.py` carry **no
literal default values** in their argparse flags — every value comes from
`asciimaze/fixed4x4/config.yaml`, which each script auto-loads via
`--config`'s own default:

```bash
python asciimaze/gemma4_train.py  --gpu 3
python asciimaze/gemma4_export.py --gpu 3
python asciimaze/gemma4_gguf.py   --gpu 3
```

The file has one top-level `train:`/`export:`/`gguf:` section per script —
**each script reads only its own section** and ignores the rest, so the
sections are fully independent (e.g. `export:` and `gguf:` each repeat their
own `gpu` / `lora_path` / `load_in_4bit` rather than sharing one). Keys are
the long flag name with dashes replaced by underscores (e.g. `--output-dir`
→ `output_dir`):

```yaml
experiment_dir: asciimaze/fixed4x4/finetune

train:
  gpu: 3
  max_seq_length: 1024
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  num_train_epochs: 2
  output_dir: ${experiment_dir}/lora
  # ...plus model/LoRA/data/training settings, see the file itself

export:
  gpu: 3
  save_merged: true
  load_in_4bit: false
  lora_path: ${experiment_dir}/lora
  merged_dir: ${experiment_dir}/merged

gguf:
  gpu: 3
  save_gguf: true
  quantization_method: F16
  load_in_4bit: false
  lora_path: ${experiment_dir}/lora
  gguf_dir: ${experiment_dir}/gguf
```

`experiment_dir` is a single top-level value substituted into any
`${experiment_dir}` in a path string before the script sees it — change it
once and `output_dir`/`lora_path`/`merged_dir`/`gguf_dir` all move together,
so training and export/GGUF can't drift onto different LoRA paths (a
previous flat-config layout made that easy to do by accident). It's plain
string substitution, not a YAML feature — `${experiment_dir}` only resolves
inside values under `train:`/`export:`/`gguf:`, not standalone.

Any flag also passed on the command line overrides its config value for
that run (e.g. `--gpu 5` trains on GPU 5 even though the file says `gpu: 3`).

> To run a different experiment, copy `asciimaze/fixed4x4/config.yaml` (e.g.
> to `myexp/config.yaml`), update `experiment_dir` and any other values, then
> pass `--config myexp/config.yaml`. Since flags no longer carry defaults in
> the scripts themselves, a custom config must define every key its
> section needs — there's no built-in fallback for a missing key.

---

## 1. Train (LoRA)

```bash
python asciimaze/gemma4_train.py --gpu 3
```

Reads `asciimaze/fixed4x4/data/{train,val}.jsonl` (defaults), adds LoRA adapters, trains
with SFTTrainer (loss only on assistant turns), and saves the adapters (~50 MB)
to `asciimaze/fixed4x4/finetune/lora` (`--output-dir` / `output_dir`). On an A40 one
run peaked at ~12.8 GB VRAM.

- **LoRA defaults are now `--lora-r 16 --lora-alpha 32`** (scale 2.0). This is
  deliberate — see the merge-loss box under step 3.
- `--num-train-epochs 2` (with no `--max-steps`) trains ~2 epochs; the script
  auto-disables `max_steps` so epochs take effect. Too many epochs (e.g. the old
  `--max-steps 2000` ≈ 10 epochs) mode-collapses the output.

> First run silently downloads the ~10 GB base model (Unsloth suppresses the
> progress bar). It's downloading, not stuck.

## 2. Export merged float16 model

```bash
python asciimaze/gemma4_export.py --gpu 3
```

Fuses the LoRA adapters (`--lora-path` / `lora_path`) into a standalone
float16 checkpoint (~10 GB) at `--merged-dir` / `merged_dir`, for
deployment (e.g. vLLM). `--no-load-in-4bit` (`load_in_4bit: false`) merges from
the full-precision base (recommended). Add `--push-merged --hub-repo <repo>
--hf-token <token>` to upload.

## 3. Convert to GGUF (llama.cpp)

```bash
python asciimaze/gemma4_gguf.py --gpu 3
```

Produces `asciimaze/fixed4x4/finetune/gguf/gemma-4-E2B-it.F16.gguf` (~8.7 GB)
plus a `Modelfile` for Ollama. The exact filename case can vary between runs —
`ls asciimaze/fixed4x4/finetune/gguf/*.gguf` to confirm.

> ### ⚠️ Merged / GGUF looks like the base model? (small-LoRA merge loss)
>
> **Symptom:** the LoRA adapter generates correct output, but the **merged**
> and **GGUF** models ignore the finetune and behave like the un-finetuned base
> (e.g. maze format is lost).
>
> **Cause — not 4-bit, not the chat template, not a bad merge call.** A
> narrow-task LoRA at low rank/scale (`r=8, alpha=8` → scale 1.0) learns a very
> small weight delta (mean `|dW|` ~1e-4). That is **below the bf16 rounding
> step** of the base weights. The adapter still works when applied *separately*
> (it runs in its own activation path), but `save_pretrained_merged` / GGUF
> **fold `dW` into `W` and round to bf16**, discarding a delta that small. We
> measured: correlation(adapter Δ, merged Δ) ≈ **0.0006**, and **~55%** of the
> adapter deltas were smaller than one bf16 ULP. So merged ≈ base + noise.
>
> **Fixes:**
> 1. **Don't merge — serve the adapter.** The unmerged `lora/` works in 4-bit
>    *and* fp16. Use PyTorch/Unsloth, **vLLM** (native LoRA), or for llama.cpp
>    convert the *adapter* with `convert_lora_to_gguf.py` and run
>    `llama-server -m <base>.gguf --lora <adapter>.gguf` (adapter stays a
>    separate path — no merge loss).
> 2. **Retrain with a bigger footprint** so the delta survives bf16 merge — the
>    train script now defaults to **`r=16, alpha=32`** (scale 2.0). This was
>    verified: merged & GGUF then reproduce the adapter's output exactly.
> 3. **Partial:** merge and keep **fp16** (not bf16) — ~8× finer mantissa
>    preserves more, but downstream Q8/Q4 GGUF re-loses it, so prefer (1)/(2).
>
> **Sanity check** that a merge actually kept the finetune — compare merged vs
> base weights against the adapter delta (correlation should be ≈1, not ≈0), or
> just generate from both and compare to the adapter's output.

## 4. Quick inference (single prompt)

```bash
python asciimaze/gemma4_inference.py --gpu 3
# or the merged model:
python asciimaze/gemma4_inference.py --model-path asciimaze/fixed4x4/finetune/merged --gpu 3
```

Loads `--model-path` (default `…/lora`), runs one generation for `--prompt`
(defaults to the maze instruction), and streams it token-by-token. Override
sampling with `--temperature/--top-p/--top-k` (default: the recommended Gemma-4
`1.0 / 0.95 / 64`), `--max-new-tokens`, or `--no-stream`.

---

## Batch generation — 100 samples

Three scripts generate N samples (default 100) of the **same** prompt, each in a
fresh "clear session" (no history carried between generations), writing all
outputs to one file with `===== Sample i/N =====` delimiters. They set
`do_sample=True`, so the samples actually vary.

### From the LoRA adapters

```bash
python asciimaze/gemma4_generate_100_samples_with_lora.py \
  --num-samples 100 --batch-size 25 --no-load-in-4bit --gpu 3
```

### From the merged float16 model

```bash
python asciimaze/gemma4_generate_100_samples_with_merged.py \
  --num-samples 100 --batch-size 25 --gpu 3
```

### From the quantized GGUF model (via llama.cpp `llama-server`)

```bash
python asciimaze/gemma4_generate_100_samples_with_gguf.py \
  --num-samples 100 --n-gpu-layers 99 --port 8090 --gpu 3
```

Spawns `llama-server` (from `~/.unsloth/llama.cpp/`), waits until healthy, sends
N stateless `/v1/chat/completions` requests, then shuts the server down. Server
logs go to `<output-file>.server.log`. Use `--port` if 8090 is taken; set
`--n-gpu-layers 0` for CPU-only.

Defaults write to `asciimaze/fixed4x4/outputs/{lora,merged,gguf}_samples.txt`.

> To generate different maze sizes, change `--prompt` (e.g. a 5×5 or 8×8
> instruction) and pick a matching `--output-file` (e.g.
> `asciimaze/fixed4x4/outputs/gguf_samples_5x5.txt`).

### Common flags (all three)

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-samples` | `100` | number of independent generations |
| `--batch-size` | `20` | samples per GPU call (`num_return_sequences`); higher = faster, more VRAM. **lora/merged only** |
| `--prompt` | maze instruction | prompt sent every iteration |
| `--output-file` | `asciimaze/fixed4x4/outputs/<kind>_samples.txt` | where all samples are written |
| `--max-new-tokens` | `256` | generation length cap |
| `--temperature` / `--top-p` / `--top-k` | `1.0` / `0.95` / `64` | Gemma-4 recommended sampling |
| `--seed` | none (`-1` for gguf) | base seed; sample *i* uses `seed+i` (reproducible) |
| `--gpu` | `0` | CUDA device index |

---

## Serve the GGUF as an API / chat UI

```bash
CUDA_VISIBLE_DEVICES=3 ~/.unsloth/llama.cpp/llama-server \
  -m asciimaze/fixed4x4/finetune/gguf/gemma-4-E2B-it.F16.gguf \
  -ngl 99 -c 4096 --host 0.0.0.0 --port 8090
```

Then open the web UI at `http://localhost:8090`, or POST to
`http://localhost:8090/v1/chat/completions` (OpenAI-compatible).

> **Note:** the bundled `~/.unsloth/llama.cpp/llama-server` may be a **CPU-only**
> build (no CUDA linked) — then `-ngl` is ignored and generation is slow
> (~10 tok/s). For fast GPU GGUF inference, build llama.cpp with `-DGGML_CUDA=ON`
> and point `--llama-server-bin` at it.

> **Disk note:** if `~` / `/` is low on space, write large artifacts
> (`--merged-dir`, `--gguf-dir`) to a roomier disk instead.
