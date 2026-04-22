from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from .inference_service import CDVAEInferenceService
    from .schemas import GenerateRequest, GenerateResponse, HealthResponse
except ImportError:  # pragma: no cover
    from inference_service import CDVAEInferenceService
    from schemas import GenerateRequest, GenerateResponse, HealthResponse


app = FastAPI(title="Crystal Generator (CDVAE) API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = CDVAEInferenceService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = service.status
    return HealthResponse(
        status="ok",
        model_loaded=status.loaded,
        checkpoint_path=status.checkpoint_path,
        device=status.device,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        structures = service.generate_structures(request.num_samples)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return GenerateResponse(structures=structures)
