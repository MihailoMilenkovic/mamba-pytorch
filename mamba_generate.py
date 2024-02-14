# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time

import torch

from transformers import AutoTokenizer

from mamba_ssm.model import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Text Generation")
parser.add_argument("--model-ckpt-dir", type=str, default="/net/llm-compiles/mmilenkovic/mamba-130m",help="Path to model checkpoint from huggingface")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

device = "cpu"
dtype = torch.float16

print(f"Loading model from {args.model_ckpt_dir}")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.load_from_ckpt(
    args.model_ckpt_dir, device=device, dtype=dtype
)
model.eval()
print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

start = time.time()

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(
        1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda"
    )
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

fn = lambda: model.generate(
    input_ids=input_ids,
    max_length=max_length,
    cg=True,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=False,
    temperature=args.temperature,
    top_k=args.topk,
    top_p=args.topp,
    min_p=args.minp,
    repetition_penalty=args.repetition_penalty,
)
out = fn()
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences)[0])

print(
    f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}"
)
print(
    f"Prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms"
)
