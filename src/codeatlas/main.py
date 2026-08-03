from fastapi import FastAPI

app = FastAPI(
    title="CodeAtlas API",
    description="AI-powered code intelligence platform",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}