# -*- coding: utf-8 -*-
"""
Gemma-4 (E2B) Text - INFERENCE script.

Split out of `gemma4_(e2b)_text.py` (the "Inference" section of the original
notebook). All settings are now argparse flags - every flag defaults to the
value already used in the original script, so running with no flags behaves
essentially as before (recommended Gemma-4 sampling: temperature=1.0,
top_p=0.95, top_k=64).

Reloads the model saved by `gemma4_train.py` and runs a single generation for
`--prompt`. `--model-path` can point at either the LoRA adapters
(`asciimaze/fixed4x4/finetune/lora`) or the merged model (`asciimaze/fixed4x4/finetune/merged`).
"""

import argparse
import os

from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Device
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA device index to run on (sets CUDA_VISIBLE_DEVICES)")

    # Model loading
    parser.add_argument("--model-path", default="asciimaze/fixed4x4/finetune/lora",
                         help="LoRA adapters (e.g. asciimaze/fixed4x4/finetune/lora) or a "
                              "merged model (e.g. asciimaze/fixed4x4/finetune/merged)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                         help="4 bit quantization to reduce memory")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--chat-template", default="gemma-4")

    # Prompt
    parser.add_argument("--prompt", default=(
        "Generate a random, valid 4x4 ASCII maze with barriers between cells. "
        "Label the columns `A`, `B`, `C`, and `D`, and label the rows `1`, `2`, "
        "`3`, and `4`. Place `S` in the starting cell and `E` in the ending cell. "
        "The entire outer boundary of the maze must be fully enclosed with barriers "
        "on all four sides: top, bottom, left, and right. There must always be at "
        "least one valid path from the starting cell to the ending cell. Return only "
        "the maze inside a monospaced code block."),
        help="User prompt to send to the model")

    # Generation (recommended Gemma-4 settings)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--no-stream", dest="stream", action="store_false", default=True,
                         help="Disable token-by-token streaming; print the full decode instead")

    return parser.parse_args()


args = parse_args()

# Must be set before importing torch/unsloth so the CUDA context is created
# on the right device.
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Route ALL Hugging Face cache (models/hub, datasets, tokens, etc.) to the
# user's home instead of the shared /mnt/Avsol cache, which has cross-user
# permission issues. Must run before importing unsloth / transformers /
# datasets, since those read these env vars at import time. HF_CACHE_DIR
# comes from .env (see .env.example), falling back to ~/.cache/huggingface.
load_dotenv()
_HF_CACHE = os.path.expanduser(os.environ.get("HF_CACHE_DIR", "~/.cache/huggingface"))
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_CACHE, "datasets")

from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_path,  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = args.max_seq_length,
    load_in_4bit = args.load_in_4bit,
)

"""### Inference
Let's run the model via Unsloth native inference! According to the `Gemma-4`
team, the recommended settings for inference are `temperature = 1.0,
top_p = 0.95, top_k = 64`."""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = args.chat_template,
)

messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : args.prompt,
    }]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,  # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

generate_kwargs = dict(
    **inputs,
    max_new_tokens = args.max_new_tokens,
    # Recommended Gemma-4 settings!
    temperature = args.temperature, top_p = args.top_p, top_k = args.top_k,
)

# Stream token-by-token (default) so you see generation live, or fall back to a
# single batch decode with --no-stream.
if args.stream:
    from transformers import TextStreamer
    outputs = model.generate(
        **generate_kwargs,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
else:
    outputs = model.generate(**generate_kwargs)
    print(tokenizer.batch_decode(outputs)[0])
