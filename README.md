# Gemma-4 (E2B) ASCII-Maze Finetune

Finetune `unsloth/gemma-4-E2B-it` on a local ASCII-maze dataset with Unsloth +
LoRA, then export / convert / run the result. Split out of the original
`gemma4_(e2b)_text.py` notebook into standalone, argparse-driven scripts.

## Repo layout

The `gemma4_*.py` scripts live at the **repo root** and are shared across
experiments. Each experiment gets its **own top-level folder** holding its data,
model artifacts, and generation outputs — e.g. `fixed4x4/` (fixed-size 4×4
mazes):

```
gemma4_train.py  gemma4_export.py  gemma4_gguf.py         # shared scripts (repo root)
gemma4_inference.py  gemma4_generate_100_samples_*.py

fixed4x4/                          # one experiment
├── data/                          # train.jsonl / val.jsonl / test.jsonl
├── experiment/
│   ├── lora/      # LoRA adapters        (gemma4_train.py)
│   ├── merged/    # merged float16 model (gemma4_export.py --save-merged)
│   └── gguf/      # GGUF + Modelfile     (gemma4_gguf.py  --save-gguf)
└── outputs/       # batch-generation results (*.txt)
```

**All path defaults now point into `fixed4x4/`** (train/val files, model dirs,
output files), so the commands below only pass what differs from the defaults —
mainly `--gpu`. Run them from the repo root. Examples use **`--gpu 3`**; change
it to a free CUDA index (`nvidia-smi`).

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

## Config file (`fixed4x4/config.yaml`)

Instead of passing every flag on the command line, `gemma4_train.py`,
`gemma4_export.py`, and `gemma4_gguf.py` all accept `--config <path>`, e.g.:

```bash
python gemma4_train.py  --config fixed4x4/config.yaml --gpu 3
python gemma4_export.py --config fixed4x4/config.yaml --gpu 3
python gemma4_gguf.py   --config fixed4x4/config.yaml --gpu 3
```

`fixed4x4/config.yaml` is **one file shared across all three scripts** — keys
are the long flag name with dashes replaced by underscores (e.g.
`--output-dir` → `output_dir`). Each script only reads the keys it recognizes
and silently ignores the rest, so training, export, and GGUF settings can all
live together:

```yaml
# --- gemma4_train.py ---
output_dir: fixed4x4/experiment/lora
max_seq_length: 1024
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
num_train_epochs: 2
gpu: 3

# --- gemma4_export.py ---
save_merged: true
load_in_4bit: false
lora_path: fixed4x4/finetune/lora
merged_dir: fixed4x4/finetune/merged

# --- gemma4_gguf.py ---
save_gguf: true
quantization_method: F16
gguf_dir: fixed4x4/finetune/gguf
# lora_path / load_in_4bit above are shared with gemma4_export.py
```

> **Note:** in this file, training writes to `fixed4x4/experiment/lora`, but
> export/GGUF read from `fixed4x4/finetune/lora` — a different path. Update
> `lora_path` (and `merged_dir` / `gguf_dir`) if you want export/GGUF to pick
> up whatever `gemma4_train.py` just produced.

Any flag also passed on the command line overrides the config file value for
that run (e.g. `--config fixed4x4/config.yaml --gpu 5` trains on GPU 5 even
though the file says `gpu: 3`). `--max-seq-length` is deliberately shared
between all three scripts, so changing it in one section changes it
everywhere.

> To run a different experiment, copy `fixed4x4/config.yaml` (e.g. to
> `myexp/config.yaml`) and point its paths at `myexp/...`, then pass
> `--config myexp/config.yaml`.

---

## 1. Train (LoRA)

```bash
python gemma4_train.py --config fixed4x4/config.yaml --gpu 3
```

<details>
<summary>Equivalent flags (no config file)</summary>

```bash
python gemma4_train.py \
  --max-seq-length 1024 \
  --per-device-train-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --num-train-epochs 2 \
  --gpu 3
```

</details>

Reads `fixed4x4/data/{train,val}.jsonl` (defaults), adds LoRA adapters, trains
with SFTTrainer (loss only on assistant turns), and saves the adapters (~50 MB)
to `fixed4x4/experiment/lora` (`--output-dir` / `output_dir`). On an A40 one
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
python gemma4_export.py --config fixed4x4/config.yaml --gpu 3
```

<details>
<summary>Equivalent flags (no config file)</summary>

```bash
python gemma4_export.py --save-merged --no-load-in-4bit --gpu 3 \
  --lora-path  fixed4x4/finetune/lora \
  --merged-dir fixed4x4/finetune/merged
```

</details>

Fuses the LoRA adapters (`--lora-path` / `lora_path`) into a standalone
float16 checkpoint (~10 GB) at `--merged-dir` / `merged_dir`, for
deployment (e.g. vLLM). `--no-load-in-4bit` (`load_in_4bit: false`) merges from
the full-precision base (recommended). Add `--push-merged --hub-repo <repo>
--hf-token <token>` to upload.

## 3. Convert to GGUF (llama.cpp)

```bash
python gemma4_gguf.py --config fixed4x4/config.yaml --gpu 3
```

<details>
<summary>Equivalent flags (no config file)</summary>

```bash
python gemma4_gguf.py --save-gguf --no-load-in-4bit --quantization-method F16 --gpu 3 \
  --lora-path fixed4x4/finetune/lora \
  --gguf-dir  fixed4x4/finetune/gguf
```

</details>

Produces `fixed4x4/finetune/gguf/gemma-4-E2B-it.F16.gguf` (~8.7 GB)
plus a `Modelfile` for Ollama. The exact filename case can vary between runs —
`ls fixed4x4/finetune/gguf/*.gguf` to confirm.

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
python gemma4_inference.py --gpu 3
# or the merged model:
python gemma4_inference.py --model-path fixed4x4/experiment/merged --gpu 3
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
python gemma4_generate_100_samples_with_lora.py \
  --num-samples 100 --batch-size 25 --no-load-in-4bit --gpu 3
```

### From the merged float16 model

```bash
python gemma4_generate_100_samples_with_merged.py \
  --num-samples 100 --batch-size 25 --gpu 3
```

### From the quantized GGUF model (via llama.cpp `llama-server`)

```bash
python gemma4_generate_100_samples_with_gguf.py \
  --num-samples 100 --n-gpu-layers 99 --port 8090 --gpu 3
```

Spawns `llama-server` (from `~/.unsloth/llama.cpp/`), waits until healthy, sends
N stateless `/v1/chat/completions` requests, then shuts the server down. Server
logs go to `<output-file>.server.log`. Use `--port` if 8090 is taken; set
`--n-gpu-layers 0` for CPU-only.

Defaults write to `fixed4x4/outputs/{lora,merged,gguf}_samples.txt`.

> To generate different maze sizes, change `--prompt` (e.g. a 5×5 or 8×8
> instruction) and pick a matching `--output-file` (e.g.
> `fixed4x4/outputs/gguf_samples_5x5.txt`).

### Common flags (all three)

| Flag | Default | Meaning |
|------|---------|---------|
| `--num-samples` | `100` | number of independent generations |
| `--batch-size` | `20` | samples per GPU call (`num_return_sequences`); higher = faster, more VRAM. **lora/merged only** |
| `--prompt` | maze instruction | prompt sent every iteration |
| `--output-file` | `fixed4x4/outputs/<kind>_samples.txt` | where all samples are written |
| `--max-new-tokens` | `256` | generation length cap |
| `--temperature` / `--top-p` / `--top-k` | `1.0` / `0.95` / `64` | Gemma-4 recommended sampling |
| `--seed` | none (`-1` for gguf) | base seed; sample *i* uses `seed+i` (reproducible) |
| `--gpu` | `0` | CUDA device index |

---

## Serve the GGUF as an API / chat UI

```bash
CUDA_VISIBLE_DEVICES=3 ~/.unsloth/llama.cpp/llama-server \
  -m fixed4x4/experiment/gguf/gemma-4-E2B-it.F16.gguf \
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
