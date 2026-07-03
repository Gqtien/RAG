from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@lru_cache(maxsize=1)
def load(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    model.eval()
    return tokenizer, model


def generate(
    prompt: str,
    model_name: str = "Qwen/Qwen3-0.6B",
    max_new_tokens: int = 256,
    max_input_tokens: int = 1024,
    think: bool = False,
) -> str:
    tokenizer, model = load(model_name)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
