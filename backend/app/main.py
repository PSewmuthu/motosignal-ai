from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MotoSignal AI API", version="0.1.0",
              description="Evidence-driven motorcycle red-light event analysis and rear-plate recognition.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173"],
                   allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.get("/")
def root():
    return {"name": "MotoSignal AI", "api": "/docs", "health": "/api/health"}
