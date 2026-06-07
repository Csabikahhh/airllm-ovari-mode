r"""
AirLLM quick-start — minimal inference example.

Tailored for an 8GB GPU (RTX 5070 Laptop). AirLLM streams the model layer
by layer, so even large models "fit" — the real cost is disk space and the
one-time download + layer-splitting step into your HuggingFace cache.

Start small (this 3B model) to confirm the whole pipeline works, then swap
MODEL_ID for something bigger once you're happy.

Run with:
    .\.venv\Scripts\Activate.ps1
    python quickstart.py
"""

import torch
from airllm import AutoModel

# --- pick your model ------------------------------------------------------
# Small + fast first test (~6GB download). Bump to a 7B/70B later.
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MAX_LENGTH = 128
MAX_NEW_TOKENS = 32

# Set compression='4bit' (needs `pip install bitsandbytes`) for ~3x speedup.
model = AutoModel.from_pretrained(MODEL_ID)  # , compression='4bit')

# --- run inference --------------------------------------------------------
prompt = ["What is the capital of the United States?"]

inputs = model.tokenizer(
    prompt,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,  # off to avoid "tokenizer has no padding token" errors
)

input_ids = inputs["input_ids"]
if torch.cuda.is_available():
    input_ids = input_ids.cuda()

output = model.generate(
    input_ids,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    return_dict_in_generate=True,
)

print("\n=== OUTPUT ===")
print(model.tokenizer.decode(output.sequences[0]))
