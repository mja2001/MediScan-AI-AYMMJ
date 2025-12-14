from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.analyze import router as analyze_router
from api.report import router as report_router
from api.history import router as history_router
from api.triage import router as triage_router
from config import settings

app = FastAPI(title="MediScan AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api/analyze")
app.include_router(report_router, prefix="/api/report")
app.include_router(history_router, prefix="/api/history")
app.include_router(triage_router, prefix="/api/triage")

@app.get("/")
def read_root():
    return {"message": "MediScan AI Backend is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
