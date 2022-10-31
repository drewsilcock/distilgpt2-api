from functools import cache
from importlib.metadata import version
import logging

from fastapi import FastAPI, Request

from .text_generation import TextGenerator

app = FastAPI()


def package() -> str:
    return __name__.split(".")[0]


@cache
def get_model() -> TextGenerator:
    logging.info("Loading DistilGPT2 model")
    return TextGenerator()


@app.on_event("startup")
async def on_startup():
    get_model()


@app.get("/")
async def health(request: Request):
    model = get_model()
    return {
        "health": "ok",
        "app_version": version(package()),
        "torch_version": version("torch"),
        "model": {
            "device": model.generator.model.device.type,
            "call_count": model.generator.call_count,
            "name_or_path": model.generator.model.name_or_path,
        },
    }


@app.get("/{prompt}")
async def generate_text(
    prompt: str,
    max_new_tokens: int = 50,
    num_return_sequences: int = 1,
) -> dict:
    model = get_model()
    sequences = model.generate(
        prompt, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences
    )
    return {"generated_sequences": sequences}
