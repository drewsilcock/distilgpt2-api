from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def generate_text(prompt: str) -> str:
    return "Todo"
