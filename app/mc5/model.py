"""Local text LLM on the AMD GPU (ROCm exposes it as torch.cuda). Loaded once,
by the resident server only."""
from __future__ import annotations

import os
import time

from . import paths

# Repo-level prompts carry ~28k characters of code plus the issue. Truncation keeps the
# END of the prompt, so a cap that is too low silently drops the issue text itself.
MAX_INPUT_TOKENS = int(os.environ.get("MC5_MAX_INPUT_TOKENS", "16000"))
# Tests only (CI runs a tiny model on CPU to exercise this code path). Never set in the image.
ALLOW_CPU = os.environ.get("MC5_ALLOW_CPU") == "1"
# Read a long prompt in slices of this many tokens (0 = all at once). Attention memory grows
# with slice x prompt instead of prompt x prompt, which is what ran small GPUs out of memory
# on 6k-token prompts. The answer is the same either way.
PREFILL_CHUNK = int(os.environ.get("MC5_PREFILL_CHUNK", "0"))


class LocalLLM:
    def __init__(self, model_dir: str = paths.MODEL_DIR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif ALLOW_CPU:
            self.device = "cpu"
        else:
            raise RuntimeError("no GPU visible to torch (ROCm)")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        # Older GPUs (Kaggle's T4, used only for practice runs) have no fast bfloat16.
        forced = os.environ.get("MC5_DTYPE", "")
        if forced in ("float16", "bfloat16", "float32"):
            dtype = getattr(torch, forced)
        extra = {}
        if os.environ.get("MC5_LOAD_4BIT") == "1":      # small practice GPUs only; never set in the image
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=dtype, local_files_only=True, attn_implementation="sdpa", **extra,
                # MC5_DEVICE_MAP=auto spreads the model over several small GPUs (practice runs only).
                device_map=os.environ.get("MC5_DEVICE_MAP") or self.device,
            ).eval()
        except Exception as e:  # noqa: BLE001 - older/odd accelerate setups
            paths.log(f"device_map load failed ({type(e).__name__}: {e}); loading on CPU then moving")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=dtype, local_files_only=True, attn_implementation="sdpa",
            ).to(self.device).eval()
        self.chat("", "Say OK.", max_new_tokens=2)          # warm up kernels before the clock starts

    def chat(self, system: str, text: str, max_new_tokens: int = 120, max_time: float | None = None) -> str:
        """max_time: seconds this call may take in all. Generation stops there and the
        partial reply is returned (complete edit blocks in it are still usable)."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True, enable_thinking=False)
        ids = enc["input_ids"][:, -MAX_INPUT_TOKENS:].to(self.device)
        t0 = time.time()
        self._t_end = None if max_time is None else t0 + max_time
        with self.torch.inference_mode():
            out = self._generate(ids, max_new_tokens)
        n_out = out.shape[1] - ids.shape[1]
        paths.log(f"llm: {ids.shape[1]} tokens in, {n_out} out (cap {max_new_tokens}), {time.time() - t0:.1f}s")
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def _time_kw(self) -> dict:
        t_end = getattr(self, "_t_end", None)
        if t_end is None:
            return {}
        left = t_end - time.time()
        if left < 0.3:
            raise TimeoutError("no time left to generate")
        return {"max_time": left}

    def _generate(self, ids, max_new_tokens: int):
        torch = self.torch
        mask = torch.ones_like(ids)
        if PREFILL_CHUNK > 0 and ids.shape[1] > PREFILL_CHUNK + 1:
            try:
                past = None
                body = ids[:, :-1]                       # the last token is left for generate()
                for s in range(0, body.shape[1], PREFILL_CHUNK):
                    step = self.model(input_ids=body[:, s:s + PREFILL_CHUNK], past_key_values=past, use_cache=True)
                    past = step.past_key_values
                    del step
                    self._time_kw()                      # raises once the budget is gone
                return self.model.generate(input_ids=ids, attention_mask=mask, past_key_values=past,
                                           max_new_tokens=max_new_tokens, do_sample=False, **self._time_kw())
            except TimeoutError:
                raise
            except Exception as e:  # noqa: BLE001 - fall back to the plain path
                paths.log(f"llm: sliced prefill failed ({type(e).__name__}: {str(e)[:200]}); reading the prompt at once")
                past = None
                if self.device != "cpu":
                    torch.cuda.empty_cache()
        return self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                   do_sample=False, **self._time_kw())
