import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models import ModelConfig
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def load(model_name: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    model.eval()
    return tokenizer, model


def generate(
    config: ModelConfig,
    prompt: str,
) -> str:
    tokenizer, model = load(config.name)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_context_length,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
    generated = out[0][inputs.input_ids.shape[1]:]
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()
