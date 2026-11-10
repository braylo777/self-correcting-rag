"""Local HF chat model exposing yes/no and multiple-choice probabilities from next-token logits."""
from __future__ import annotations

import torch


class HFLLM:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16 if self.device == "mps"
                                                          else torch.float32).to(self.device).eval()
        self.calls = 0

    def _ids(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def _next_logits(self, prompt):
        self.calls += 1
        return self.model(self._ids(prompt)).logits[0, -1].float().cpu()

    def _tid(self, s):
        return self.tok.encode(s, add_special_tokens=False)[0]

    def choice_probs(self, prompt, options):
        lg = self._next_logits(prompt)
        ids = [self._tid(o) for o in options]
        return torch.softmax(lg[ids], 0).tolist()

    def yes_prob(self, prompt):
        lg = self._next_logits(prompt)
        y = torch.logsumexp(lg[[self._tid("yes"), self._tid("Yes")]], 0)
        n = torch.logsumexp(lg[[self._tid("no"), self._tid("No")]], 0)
        return float(torch.sigmoid(y - n))

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=32):
        self.calls += 1
        ids = self._ids(prompt)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
