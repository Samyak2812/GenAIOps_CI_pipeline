# LLM wrapper: supports local HF generation pipeline and OpenAI
from typing import Optional
import os

class LLM:
    def __init__(self, backend: str = "local", hf_model: str = "gpt2", openai_model: str = "gpt-4o-mini"):
        self.backend = backend
        self.hf_model = hf_model
        self.openai_model = openai_model
        self.generator = None
        if backend == "local":
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                # load model lazily
                self.generator = None
            except Exception as e:
                print("Transformers may not be available:", e)

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        if self.backend == "openai":
            import openai
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            openai.api_key = key
            resp = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            # extract content (works for ChatCompletion-like responses)
            choices = resp.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                return text
            return ""
        else:
            # local HF pipeline (lazily create)
            if self.generator is None:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                # load small model for demo; for production use a suitable model
                model = AutoModelForCausalLM.from_pretrained(self.hf_model)
                tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if os.getenv("USE_GPU","0")=="1" else -1)
            out = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=(temperature>0.0), temperature=temperature)
            # pipeline returns list of dicts with 'generated_text'
            if out and isinstance(out, list):
                return out[0].get("generated_text","")
            return str(out)
