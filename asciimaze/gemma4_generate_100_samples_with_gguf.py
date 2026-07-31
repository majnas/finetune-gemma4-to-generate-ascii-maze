# -*- coding: utf-8 -*-
"""
Gemma-4 (E2B) - generate N samples from the quantized GGUF model via llama.cpp.

The unsloth llama.cpp build ships `llama-server` (no `llama-cli`), so this
script spawns `llama-server` on the given GGUF file, waits until it reports
healthy, then runs the SAME prompt N times (default 100) as independent,
stateless `/v1/chat/completions` requests. Each request carries ONLY the user
prompt (a "clear session" - no conversation history), so the N outputs are
independent samples. Every output is written to a single output file (one
delimited block per sample, flushed after each). The server is shut down at the
end.

Sampling uses the recommended Gemma-4 settings (temperature=1.0, top_p=0.95,
top_k=64). llama-server applies the Gemma-4 chat template embedded in the GGUF.
Only stdlib is used (urllib + subprocess) - no extra Python packages needed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

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
                         help="CUDA device index for the server (sets CUDA_VISIBLE_DEVICES)")

    # Model / server
    parser.add_argument("--gguf-path",
                         default="asciimaze/fixed4x4/finetune/gguf/gemma-4-E2B-it.F16.gguf",
                         help="Path to the .gguf file saved by gemma4_gguf.py")
    parser.add_argument("--llama-server-bin",
                         default=os.path.expanduser("~/.unsloth/llama.cpp/llama-server"),
                         help="Path to the llama-server binary")
    parser.add_argument("--n-gpu-layers", type=int, default=99,
                         help="Layers to offload to GPU (-ngl). 0 = CPU only. Ignored "
                              "by CPU-only llama.cpp builds.")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Server context size (-c)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--startup-timeout", type=int, default=180,
                         help="Seconds to wait for the server to become healthy")

    # Sampling loop
    parser.add_argument("--num-samples", type=int, default=100,
                         help="How many independent generations to produce")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                         help="User prompt to send to the model (default: the maze "
                              "instruction used for finetuning)")
    parser.add_argument("--output-file", default="asciimaze/fixed4x4/outputs/gguf_samples.txt",
                         help="File to write all generations to")

    # Generation (recommended Gemma-4 settings)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=-1,
                         help="Base seed; sample i uses seed+i for reproducibility. "
                              "-1 (default) = random each request.")

    return parser.parse_args()


def wait_until_healthy(base_url, proc, timeout):
    """Poll llama-server's /health until it returns 200, or raise on timeout /
    server exit."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}). "
                f"Check the server log.")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # not up yet
        time.sleep(1.0)
    raise TimeoutError(f"llama-server did not become healthy within {timeout}s")


def chat_once(base_url, prompt, args, seed):
    """One independent, stateless chat request -> generated text."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def main():
    args = parse_args()

    if not os.path.isfile(args.gguf_path):
        sys.exit(f"GGUF file not found: {args.gguf_path}")
    if not os.path.isfile(args.llama_server_bin):
        sys.exit(f"llama-server binary not found: {args.llama_server_bin}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    server_log_path = args.output_file + ".server.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        args.llama_server_bin,
        "-m", args.gguf_path,
        "-ngl", str(args.n_gpu_layers),
        "-c", str(args.ctx_size),
        "--host", args.host,
        "--port", str(args.port),
    ]
    base_url = f"http://{args.host}:{args.port}"

    print(f"Starting llama-server (log -> {server_log_path}) ...")
    with open(server_log_path, "w") as server_log:
        proc = subprocess.Popen(cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            wait_until_healthy(base_url, proc, args.startup_timeout)
            print(f"Server ready at {base_url}. Generating {args.num_samples} samples ...")

            with open(args.output_file, "w") as f:
                f.write(f"# gguf_path      : {args.gguf_path}\n")
                f.write(f"# num_samples    : {args.num_samples}\n")
                f.write(f"# sampling       : temperature={args.temperature} "
                        f"top_p={args.top_p} top_k={args.top_k} "
                        f"max_new_tokens={args.max_new_tokens}\n")
                f.write(f"# seed           : {args.seed}\n")
                f.write(f"# prompt         : {args.prompt}\n")
                f.flush()

                for i in range(args.num_samples):
                    seed = args.seed + i if args.seed >= 0 else -1
                    text = chat_once(base_url, args.prompt, args, seed)
                    f.write(f"\n===== Sample {i + 1}/{args.num_samples} =====\n{text}\n")
                    f.flush()
                    print(f"[{i + 1}/{args.num_samples}] generated ({len(text)} chars)")

            print(f"\nDone. {args.num_samples} samples written to {args.output_file}")
        finally:
            print("Shutting down llama-server ...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
