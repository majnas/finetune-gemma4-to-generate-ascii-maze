#!/usr/bin/env python3
"""Generate solver predictions from test JSONL using native or GGUF models.

Native example (LoRA or merged):
    python -m asciimaze.fixed4x4_solver.generate_predictions \
      --backend native --model-path asciimaze/fixed4x4_solver/finetune/lora

GGUF example (with llama-server already running):
    python -m asciimaze.fixed4x4_solver.generate_predictions \
      --backend openai --base-url http://localhost:8090
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from .paths import BASE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "openai"), required=True)
    parser.add_argument("--input-file", type=Path, default=BASE_DIR / "data/test.jsonl")
    parser.add_argument("--output-file", type=Path, default=BASE_DIR / "outputs/predictions.jsonl")
    parser.add_argument("--model-path", default=str(BASE_DIR / "finetune/lora"))
    parser.add_argument("--base-url", default="http://localhost:8090")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_records(path: Path, limit: int | None) -> list[dict]:
    with path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:limit] if limit is not None else records


def openai_predict(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"].strip()


def native_predictor(args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    load_dotenv()
    cache = os.path.expanduser(os.environ.get("HF_CACHE_DIR", "~/.cache/huggingface"))
    os.environ["HF_HOME"] = cache
    os.environ["HF_HUB_CACHE"] = os.path.join(cache, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache, "datasets")

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    FastModel.for_inference(model)

    def predict(prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to("cuda")
        prompt_length = inputs["input_ids"].shape[1]
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )[0]
        return tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()

    return predict


def main() -> None:
    args = parse_args()
    records = load_records(args.input_file, args.limit)
    if args.backend == "native":
        predict = native_predictor(args)
    else:
        predict = lambda prompt: openai_predict(
            args.base_url, args.model, prompt, args.max_new_tokens
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as handle:
        for index, record in enumerate(records, 1):
            prompt = record["conversations"][0]["content"]
            expected = record["conversations"][1]["content"]
            result = {
                "meta": record["meta"],
                "expected": expected,
                "prediction": predict(prompt),
            }
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"[{index}/{len(records)}] generated")
    print(f"Predictions written to {args.output_file}")


if __name__ == "__main__":
    main()
