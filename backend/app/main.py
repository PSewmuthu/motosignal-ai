from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.db.session import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MotoSignal AI API", version="1.0.0",
              description="Evidence-driven motorcycle red-light event analysis and rear-plate recognition.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173"],
                   allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
app.include_router(router)

(settings.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
app.mount("/evidence",
          StaticFiles(directory=str(settings.data_dir / "jobs")), name="evidence")


@app.get("/")
def root():
    return {"name": "MotoSignal AI", "api": "/docs", "health": "/api/health"}
