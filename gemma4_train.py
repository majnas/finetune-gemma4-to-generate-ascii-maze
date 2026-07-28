# -*- coding: utf-8 -*-
"""
Gemma-4 (E2B) Text - TRAINING script.

Split out of `gemma4_(e2b)_text.py` (the "Data Prep" + "Train" sections of the
original notebook), then pointed at a local ASCII-maze dataset instead of the
original notebook's FineTome-100k. All training parameters are now argparse
flags - every flag defaults to the value already used, so running with no
flags behaves exactly as before.

Loads the base model, adds LoRA adapters, prepares the ASCII-maze dataset
(`data/train.jsonl` + `data/val.jsonl`), trains with SFTTrainer, and saves
the resulting LoRA adapters locally to `fixed4x4/experiment/lora/`.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth  # Do this in local & cloud setups
# else:
#     import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
#     xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
#     !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
#     !pip install --no-deps --upgrade "torchao>=0.16.0"
# !pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
# !pip install torchcodec
# import torch; torch._dynamo.config.recompile_limit = 64;

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install --no-deps --upgrade timm # For Gemma 4 vision/audio

"""### Unsloth

`FastModel` supports loading nearly any model now! This includes Vision and Text models!
"""

import argparse
import os

import yaml

# Route ALL Hugging Face cache (models/hub, datasets, tokens, etc.) to the
# user's home instead of the shared /mnt/Avsol cache, which has cross-user
# permission issues. Must run before importing unsloth / transformers /
# datasets, since those read these env vars at import time.
_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_CACHE, "datasets")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Config file
    parser.add_argument("--config", default=None,
                         help="Path to a YAML file of argument defaults (e.g. fixed4x4/config.yaml). "
                              "Keys match the long flag names with dashes replaced by underscores "
                              "(e.g. output_dir, num_train_epochs). Any flag also passed on the "
                              "command line overrides the config file value.")

    # Device
    parser.add_argument("--gpu", type=int, default=0,
                         help="CUDA device index to run on (sets CUDA_VISIBLE_DEVICES)")

    # Model loading
    parser.add_argument("--model-name", default="unsloth/gemma-4-E2B-it",
                         help="More models at https://huggingface.co/unsloth")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=False,
                         help="4 bit quantization to reduce memory")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--full-finetuning", action="store_true", default=False)

    # LoRA
    parser.add_argument("--finetune-vision-layers", action="store_true", default=False,
                         help="Turn off for just text!")
    parser.add_argument("--finetune-language-layers", action="store_true", default=True)
    parser.add_argument("--no-finetune-language-layers", dest="finetune_language_layers", action="store_false")
    parser.add_argument("--finetune-attention-modules", action="store_true", default=True)
    parser.add_argument("--no-finetune-attention-modules", dest="finetune_attention_modules", action="store_false")
    parser.add_argument("--finetune-mlp-modules", action="store_true", default=True)
    parser.add_argument("--no-finetune-mlp-modules", dest="finetune_mlp_modules", action="store_false")
    # NOTE on r / alpha defaults (raised from 8 / 8):
    # With r=8, alpha=8 (scale = alpha/r = 1.0) the learned weight delta for a
    # narrow task like this is TINY - mean |dW| ~1e-4, which is BELOW the bf16
    # rounding step of the base weights. The adapter still works when applied
    # separately (LoRA path computes it in its own activation path), but
    # `save_pretrained_merged` / GGUF fold dW into W and round to bf16, which
    # discards a finetune that small -> merged/GGUF models revert to ~base
    # behavior. r=16 + alpha=32 (scale 2.0) gives a larger weight-space
    # footprint that survives the merge. If you only ever serve the unmerged
    # LoRA adapter, the old 8/8 was already fine.
    parser.add_argument("--lora-r", type=int, default=16,
                         help="Larger = higher accuracy / bigger merge-safe delta, but might overfit")
    parser.add_argument("--lora-alpha", type=int, default=32,
                         help="Scale = alpha/r. >=2*r helps the merged/GGUF model keep the "
                              "finetune (small deltas are lost to bf16 rounding on merge)")
    parser.add_argument("--lora-dropout", type=float, default=0)
    parser.add_argument("--lora-bias", default="none")
    parser.add_argument("--random-state", type=int, default=3407)

    # Data
    parser.add_argument("--chat-template", default="gemma-4")
    parser.add_argument("--train-file", default="fixed4x4/data/train.jsonl")
    parser.add_argument("--val-file", default="fixed4x4/data/val.jsonl")

    # SFTConfig / training
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                         help="Use GA to mimic batch size!")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-train-epochs", type=float, default=None,
                         help="Set this for a full training run. HF's Trainer always lets "
                              "--max-steps override --num-train-epochs when max_steps > 0, "
                              "so if you don't also pass --max-steps explicitly, it's "
                              "automatically set to -1 (disabled) here so epochs take effect.")
    parser.add_argument("--max-steps", type=int, default=None,
                         help="Defaults to 60 if --num-train-epochs is not set, else -1 "
                              "(disabled, so --num-train-epochs takes effect)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                         help="Reduce to 2e-5 for long training runs")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none",
                         help="Use TrackIO/WandB etc")

    # Output
    parser.add_argument("--output-dir", default="fixed4x4/experiment/lora",
                         help="Where to save the trained LoRA adapters")

    # Load --config first so its values become the new argparse defaults;
    # any flag explicitly passed on the command line still overrides it. The
    # same YAML file is shared across the gemma4_*.py scripts, so keys this
    # script doesn't recognize (e.g. export-only --lora-path) are ignored
    # rather than treated as an error.
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

from unsloth import FastModel
import torch

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_name,
    dtype = None, # None for auto detection
    max_seq_length = args.max_seq_length, # Choose any for long context!
    load_in_4bit = args.load_in_4bit,
    full_finetuning = args.full_finetuning, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

"""Let's finetune Gemma 4!

You can finetune the vision and text parts for now through selection - the audio part can also be finetuned - we're working to make it selectable as well!

We now add LoRA adapters so we only need to update a small amount of parameters!
"""

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = args.finetune_vision_layers,
    finetune_language_layers   = args.finetune_language_layers,
    finetune_attention_modules = args.finetune_attention_modules, # Attention good for GRPO
    finetune_mlp_modules       = args.finetune_mlp_modules,

    r = args.lora_r,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout,
    bias = args.lora_bias,
    random_state = args.random_state,
)

"""<a name="Data"></a>
### Data Prep
We now use the `Gemma-4` format for conversation style finetunes. Gemma-4 renders multi turn conversations like below:

```
<bos><|turn>user
Hello<turn|>
<|turn>model
Hey there!<turn|>
```
We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3, gemma-4` and more.
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = args.chat_template,
)

"""Load the local ASCII-maze train/validation splits (`data/*.jsonl`, already
reformatted to the `conversations` field `standardize_data_formats` expects)."""

from datasets import load_dataset
dataset = load_dataset("json", data_files = args.train_file, split = "train")
eval_dataset = load_dataset("json", data_files = args.val_file, split = "train")

"""We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!"""

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)
eval_dataset = standardize_data_formats(eval_dataset)

"""Let's see how row 100 looks like!"""

dataset[100]

"""We now have to apply the chat template for `Gemma-4` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one."""

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)

"""Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one."""

dataset[100]["text"]

"""<a name="Train"></a>
### Train the model
Now let's train our model. We do 60 steps by default to speed things up, but you can pass `--num-train-epochs 1` for a full run.
"""

from trl import SFTTrainer, SFTConfig

# HF's Trainer always lets max_steps override num_train_epochs when max_steps
# > 0 - so if the user asked for epoch-based training and didn't also pin
# --max-steps explicitly, disable it (-1) so --num-train-epochs actually takes
# effect. With neither flag passed, this reproduces the original max_steps=60.
if args.max_steps is not None:
    max_steps = args.max_steps
elif args.num_train_epochs is not None:
    max_steps = -1
else:
    max_steps = 60

sft_config_kwargs = dict(
    dataset_text_field = "text",
    per_device_train_batch_size = args.per_device_train_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation_steps, # Use GA to mimic batch size!
    warmup_steps = args.warmup_steps,
    max_steps = max_steps,
    learning_rate = args.learning_rate, # Reduce to 2e-5 for long training runs
    logging_steps = args.logging_steps,
    optim = args.optim,
    weight_decay = args.weight_decay,
    lr_scheduler_type = args.lr_scheduler_type,
    seed = args.seed,
    report_to = args.report_to, # Use TrackIO/WandB etc
)
if args.num_train_epochs is not None: # Set this for 1 full training run.
    sft_config_kwargs["num_train_epochs"] = args.num_train_epochs

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset,
    args = SFTConfig(**sft_config_kwargs),
)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes! Unsloth now auto-detects the instruction and response parts from the tokenizer's chat template, so we don't need to pass `instruction_part` and `response_part` anymore. You can still pass them explicitly if you use a custom chat template."""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(trainer)

"""Let's verify masking the instruction part is done! Let's print the 100th row again.  Notice how the sample only has a single `<bos>` as expected!"""

tokenizer.decode(trainer.train_dataset[100]["input_ids"])

"""Now let's print the masked out example - you should see only the answer is present:"""

tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""# Let's train the model!

To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
"""

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""<a name="Save"></a>
### Saving finetuned LoRA adapters
To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, see `gemma4_export.py` / `gemma4_gguf.py`.
"""

model.save_pretrained(args.output_dir)  # Local saving
tokenizer.save_pretrained(args.output_dir)
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
