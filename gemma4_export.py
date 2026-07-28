# -*- coding: utf-8 -*-
"""
Gemma-4 (E2B) Text - EXPORT: save/push merged 16bit model.

Split out of `gemma4_(e2b)_text.py` ("Saving to float16 for VLLM" section).
Text-only finetune workflow - the multimodal (vision/audio) demo from the
original notebook was dropped, since it's unrelated to exporting a text LoRA
finetune. No settings were changed from the original script - the original
notebook's `if False` toggles are now argparse flags (all off by default,
matching the original's `if False`).

Reloads the LoRA adapters saved by `gemma4_train.py` (from
`fixed4x4/experiment/lora/`) and merges them into a full float16 model, for
deployment (e.g. with VLLM). Defaults to writing into
`fixed4x4/experiment/merged/`, keeping all finetune artifacts (lora / merged /
gguf) nested under one `fixed4x4/experiment/` project folder.
"""

import argparse
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                         help="Path to a YAML file of argument defaults (e.g. fixed4x4/config.yaml). "
                              "Keys match the long flag names with dashes replaced by underscores "
                              "(e.g. lora_path, merged_dir). Any flag also passed on the command "
                              "line overrides the config file value. Keys this script doesn't "
                              "recognize (e.g. train-only keys) are ignored.")
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA device index to run on (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--lora-path", default="fixed4x4/experiment/lora",
                         help="Path to the LoRA adapters saved by gemma4_train.py")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")

    parser.add_argument("--save-merged", action="store_true",
                         help="Save the merged float16 model locally (originally `if False` in the notebook)")
    parser.add_argument("--merged-dir", default="fixed4x4/experiment/merged",
                         help="Local output folder for the merged float16 model")

    parser.add_argument("--push-merged", action="store_true",
                         help="Push the merged float16 model to the Hugging Face Hub (originally `if False` in the notebook)")
    parser.add_argument("--hub-repo", default="HF_ACCOUNT/gemma_4_finetune",
                         help="Hub repo id to push the merged model to")
    parser.add_argument("--hf-token", default="YOUR_HF_TOKEN",
                         help="Hugging Face token for pushing to the Hub")

    # Load --config first so its values become the new argparse defaults;
    # any flag explicitly passed on the command line still overrides it.
    known_args, _ = parser.parse_known_args()
    if known_args.config is not None:
        with open(known_args.config) as f:
            config = yaml.safe_load(f) or {}
        valid_dests = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in config.items() if k in valid_dests})

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

model, tokenizer = FastModel.from_pretrained(
    model_name = args.lora_path, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = args.max_seq_length,
    load_in_4bit = args.load_in_4bit,
)

"""### Saving to float16 for VLLM

We also support saving to `float16` directly for deployment! We save it in the folder `fixed4x4/experiment/merged`. Pass `--save-merged` to let it run!
"""

if args.save_merged: # Change to True to save finetune!
    model.save_pretrained_merged(args.merged_dir, tokenizer)

"""If you want to upload / push to your Hugging Face account, pass `--push-merged` and set `--hub-repo` / `--hf-token`!"""

if args.push_merged: # Change to True to upload finetune
    model.push_to_hub_merged(
        args.hub_repo, tokenizer,
        token = args.hf_token
    )
