import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .workflow import build_claim_graph

app = FastAPI(title="Claim Processing Pipeline", version="1.0.0")
claim_graph = build_claim_graph()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="claim_upload_"))
    pdf_path = tmp_dir / "input.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    initial_state = {"claim_id": claim_id, "pdf_path": str(pdf_path)}
    result = claim_graph.invoke(initial_state)
    return result.get("final_result", {})

