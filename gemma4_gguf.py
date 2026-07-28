# -*- coding: utf-8 -*-
"""
Gemma-4 (E2B) Text - GGUF: convert/push to GGUF for llama.cpp.

Split out of `gemma4_(e2b)_text.py` ("GGUF / llama.cpp Conversion" section).
Text-only finetune workflow - the multimodal (vision/audio) demo from the
original notebook was dropped, since it's unrelated to GGUF conversion of a
text LoRA finetune. No settings were changed from the original script - the
original notebook's `if False` toggles are now argparse flags (all off by
default, matching the original's `if False`).

Reloads the LoRA adapters saved by `gemma4_train.py` (from
`fixed4x4/experiment/lora/`) and converts to GGUF for use with llama.cpp.
Unsloth's `save_pretrained_gguf` hardcodes its GGUF output to
`f"{save_directory}_gguf"` (see `unsloth/save.py`, no parameter controls
this) - so after conversion we move those files into `--gguf-dir` itself.
Defaults to `fixed4x4/experiment/gguf/`, keeping all finetune artifacts
(lora / merged / gguf) nested under one `fixed4x4/experiment/` project folder.
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                         help="Path to a YAML file of argument defaults (e.g. fixed4x4/config.yaml). "
                              "Keys match the long flag names with dashes replaced by underscores "
                              "(e.g. lora_path, gguf_dir). Any flag also passed on the command "
                              "line overrides the config file value. Keys this script doesn't "
                              "recognize (e.g. train-only keys) are ignored.")
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA device index to run on (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--lora-path", default="fixed4x4/experiment/lora",
                         help="Path to the LoRA adapters saved by gemma4_train.py")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")

    parser.add_argument("--save-gguf", action="store_true",
                         help="Save the model as GGUF locally (originally `if False` in the notebook)")
    parser.add_argument("--gguf-dir", default="fixed4x4/experiment/gguf",
                         help="Local output folder for the GGUF files (merged 16bit intermediate + .gguf + Modelfile)")
    parser.add_argument("--quantization-method", default="Q8_0",
                         help="Only Q8_0, BF16, F16 supported")

    parser.add_argument("--push-gguf", action="store_true",
                         help="Push the GGUF model to the Hugging Face Hub (originally `if False` in the notebook)")
    parser.add_argument("--hub-repo", default="HF_ACCOUNT/gemma_4_finetune",
                         help="Hub repo id to push the GGUF model to")
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

"""### GGUF / llama.cpp Conversion
To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!
"""

if args.save_gguf: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        args.gguf_dir,
        tokenizer,
        quantization_method = args.quantization_method, # For now only Q8_0, BF16, F16 supported
    )

    # Unsloth always writes GGUF output to f"{args.gguf_dir}_gguf" - merge it
    # into args.gguf_dir so all files end up in one folder.
    gguf_output_dir = Path(f"{args.gguf_dir}_gguf")
    if gguf_output_dir.is_dir():
        target_dir = Path(args.gguf_dir)
        for item in gguf_output_dir.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        gguf_output_dir.rmdir()

"""Likewise, if you want to instead push to GGUF to your Hugging Face account, pass `--push-gguf` and set `--hub-repo` / `--hf-token`!"""

if args.push_gguf: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        args.hub_repo,
        tokenizer,
        quantization_method = args.quantization_method, # Only Q8_0, BF16, F16 supported
        token = args.hf_token,
    )

"""Now, use the `gemma-4-finetune.gguf` file or `gemma-4-finetune-Q4_K_M.gguf` file in llama.cpp."""
