# -*- coding: utf-8 -*-
"""
Gemma-4 (E2B) - generate N samples from the finetuned LoRA adapters.

Loads the LoRA adapters saved by `gemma4_train.py` (base model + adapter), then
runs the SAME prompt N times (default 100). Each generation is an independent,
"clear session": the prompt is re-encoded fresh every time with NO conversation
history carried over between generations, so the N outputs are independent
samples. Every output is written to a single output file (one delimited block
per sample, flushed after each so progress survives an interrupt).

Sampling uses the recommended Gemma-4 settings (temperature=1.0, top_p=0.95,
top_k=64) with `do_sample=True` so each of the N generations differs.
"""

import argparse
import os

# The exact instruction the maze model was finetuned on - used as the default
# prompt so this script reproduces the training task out of the box.
DEFAULT_PROMPT = (
    "Generate a random, valid 4x4 ASCII maze with barriers between cells. "
    "Label the columns `A`, `B`, `C`, and `D`, and label the rows `1`, `2`, "
    "`3`, and `4`. Place `S` in the starting cell and `E` in the ending cell. "
    "The entire outer boundary of the maze must be fully enclosed with barriers "
    "on all four sides: top, bottom, left, and right. There must always be at "
    "least one valid path from the starting cell to the ending cell. Return only "
    "the maze inside a monospaced code block."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Device
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA device index to run on (sets CUDA_VISIBLE_DEVICES)")

    # Model loading
    parser.add_argument("--model-path", default="fixed4x4/experiment/lora",
                         help="Path to the LoRA adapters saved by gemma4_train.py")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                         help="4 bit quantization to reduce memory")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--chat-template", default="gemma-4")

    # Sampling loop
    parser.add_argument("--num-samples", type=int, default=100,
                         help="How many independent generations to produce")
    parser.add_argument("--batch-size", type=int, default=20,
                         help="Samples generated per GPU call (num_return_sequences). "
                              "Higher = faster but more VRAM. Lower it if you OOM.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                         help="User prompt to send to the model (default: the maze "
                              "instruction used for finetuning)")
    parser.add_argument("--output-file", default="fixed4x4/outputs/lora_samples.txt",
                         help="File to write all generations to")

    # Generation (recommended Gemma-4 settings)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None,
                         help="Base seed; sample i uses seed+i for reproducibility. "
                              "Omit for nondeterministic sampling.")

    return parser.parse_args()


args = parse_args()

# Must be set before importing torch/unsloth so the CUDA context is created
# on the right device.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Route ALL Hugging Face cache (models/hub, datasets, tokens, etc.) to the
# user's home instead of the shared /mnt/Avsol cache, which has cross-user
# permission issues. Must run before importing unsloth / transformers /
# datasets, since those read these env vars at import time.
_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_CACHE, "datasets")

from unsloth import FastModel
import torch
from transformers import set_seed

model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_path,  # LoRA adapters saved by gemma4_train.py
    max_seq_length = args.max_seq_length,
    load_in_4bit = args.load_in_4bit,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = args.chat_template,
)

# Enable Unsloth's optimized inference path (~2x faster generation).
try:
    FastModel.for_inference(model)
except Exception:
    pass


def generate_batch(prompt, n):
    """Generate `n` independent samples for `prompt` in a single batched GPU
    call (num_return_sequences=n) - far faster than n separate calls. All
    samples start from the same fresh prompt with no shared history, so each is
    an independent clear-session generation. Returns a list of `n` texts with
    the prompt stripped off."""
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,  # Must add for generation
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        num_return_sequences = n,  # n samples in one batched pass
        max_new_tokens = args.max_new_tokens,
        do_sample = True,  # REQUIRED so temperature/top_p/top_k actually sample
        temperature = args.temperature,
        top_p = args.top_p,
        top_k = args.top_k,
    )
    # Keep only the tokens generated after the prompt, for each sequence.
    return [tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
            for o in outputs]


os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

with open(args.output_file, "w") as f:
    f.write(f"# model_path     : {args.model_path}\n")
    f.write(f"# num_samples    : {args.num_samples}\n")
    f.write(f"# batch_size     : {args.batch_size}\n")
    f.write(f"# sampling       : temperature={args.temperature} top_p={args.top_p} "
            f"top_k={args.top_k} max_new_tokens={args.max_new_tokens}\n")
    f.write(f"# seed           : {args.seed}\n")
    f.write(f"# prompt         : {args.prompt}\n")
    f.flush()

    done = 0
    batch_idx = 0
    while done < args.num_samples:
        n = min(args.batch_size, args.num_samples - done)
        if args.seed is not None:
            set_seed(args.seed + batch_idx)  # reproducible per-batch seed
        for text in generate_batch(args.prompt, n):
            done += 1
            f.write(f"\n===== Sample {done}/{args.num_samples} =====\n{text}\n")
        f.flush()
        batch_idx += 1
        print(f"[{done}/{args.num_samples}] generated")

print(f"\nDone. {args.num_samples} samples written to {args.output_file}")
