"""
main.py — FastAPI backend for the CDVAE Crystal Generator web app.

Endpoints:
  POST /upload         — Upload a CSV dataset, get a 5-row preview
  POST /train          — Start VAE training (background thread)
  GET  /train-status   — Poll current training epoch / loss / done
  POST /generate       — Generate a crystal conditioned on antigravity_score
  POST /validate       — Validate a CIF structure (distance / density / volume)
  GET  /generate-multiple — Generate 5 diverse samples at once

Run with:
  uvicorn main:app --reload --port 8000
"""

import io
import os
import threading
import time
import pickle
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from model   import CrystalVAE, vae_loss
from dataset import extract_features, get_preview
from generate import params_to_cif

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="CDVAE Crystal Generator API", version="1.0")

# Allow requests from the React dev server (port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global shared state ───────────────────────────────────────────────────────
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

MODEL_PATH  = Path("cdvae.pth")
SCALER_PATH = Path("scaler.pkl")

# Training state (thread-safe via a dict + lock)
train_state: dict = {
    "running": False,
    "epoch": 0,
    "total_epochs": 300,
    "loss": 0.0,
    "recon_loss": 0.0,
    "kl_loss": 0.0,
    "done": True,
    "logs": [],
    "history": [],        # [{epoch, loss}] for loss chart
    "error": None,
}
train_lock = threading.Lock()

# In-memory model + scaler (loaded once trained or from disk)
_model:  Optional[CrystalVAE]      = None
_scaler: Optional[StandardScaler]  = None


# ── Helper: load model if saved ───────────────────────────────────────────────
def _try_load():
    global _model, _scaler
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        m = CrystalVAE()
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)


_try_load()


# ── POST /upload ──────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accept a CSV, save it, return a 5-row preview."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    contents = await file.read()
    save_path = UPLOADS_DIR / "train.csv"
    save_path.write_bytes(contents)

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Basic validation
    required = {"cif", "energy_per_atom"}
    cols_lower = {c.lower() for c in df.columns}
    missing = [r for r in required if r not in cols_lower]
    if missing:
        raise HTTPException(400, f"CSV is missing required columns: {missing}")

    preview = get_preview(df)
    return {
        "message": "Dataset uploaded successfully.",
        "rows": len(df),
        "columns": df.columns.tolist(),
        "preview": preview,
    }


# ── Training thread ───────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    epochs: int = 300
    lr: float   = 0.001
    latent_dim: int = 4


def _train_thread(epochs: int, lr: float, latent_dim: int):
    global _model, _scaler

    csv_path = UPLOADS_DIR / "train.csv"
    if not csv_path.exists():
        with train_lock:
            train_state["error"] = "No dataset uploaded. Please upload a CSV first."
            train_state["done"]  = True
            train_state["running"] = False
        return

    try:
        df = pd.read_csv(csv_path)

        with train_lock:
            train_state["logs"].append("Extracting features from CIF strings…")

        X, y, errs = extract_features(df)

        with train_lock:
            train_state["logs"].append(
                f"Features extracted: {X.shape[0]} structures, {errs} skipped."
            )

        # Scale
        scaler = StandardScaler()
        X_sc  = scaler.fit_transform(X)
        X_t   = torch.tensor(X_sc, dtype=torch.float32)

        # Model
        model     = CrystalVAE(input_dim=8, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            recon, mu, logvar = model(X_t)
            loss, r_loss, k_loss = vae_loss(recon, X_t, mu, logvar)
            loss.backward()
            optimizer.step()

            ep_data = {
                "epoch": epoch,
                "loss":  round(loss.item(), 6),
                "recon": round(r_loss, 6),
                "kl":    round(k_loss, 6),
            }
            history.append({"epoch": epoch, "loss": loss.item()})

            with train_lock:
                train_state["epoch"]      = epoch
                train_state["loss"]       = ep_data["loss"]
                train_state["recon_loss"] = ep_data["recon"]
                train_state["kl_loss"]    = ep_data["kl"]
                train_state["history"]    = history

                # Log every 10 epochs
                if epoch % 10 == 0:
                    train_state["logs"].append(
                        f"Epoch {epoch}/{epochs} | Loss={ep_data['loss']:.4f} "
                        f"Recon={ep_data['recon']:.4f} KL={ep_data['kl']:.4f}"
                    )

        # Save model + scaler
        torch.save(model.state_dict(), MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        _model  = model
        _scaler = scaler

        with train_lock:
            train_state["done"]    = True
            train_state["running"] = False
            train_state["logs"].append("✅ Training complete. Model saved as cdvae.pth")

    except Exception as exc:
        with train_lock:
            train_state["error"]   = str(exc)
            train_state["done"]    = True
            train_state["running"] = False
            train_state["logs"].append(f"❌ Error: {exc}")


# ── POST /train ───────────────────────────────────────────────────────────────
@app.post("/train")
async def start_training(req: TrainRequest = TrainRequest()):
    """Start training in a background thread."""
    with train_lock:
        if train_state["running"]:
            return {"message": "Training already in progress."}
        train_state.update({
            "running": True,
            "done": False,
            "epoch": 0,
            "total_epochs": req.epochs,
            "loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "logs": ["🚀 Training started…"],
            "history": [],
            "error": None,
        })

    t = threading.Thread(
        target=_train_thread,
        args=(req.epochs, req.lr, req.latent_dim),
        daemon=True,
    )
    t.start()
    return {"message": "Training started.", "epochs": req.epochs}


# ── GET /train-status ─────────────────────────────────────────────────────────
@app.get("/train-status")
def training_status():
    """Poll this endpoint for live training progress."""
    with train_lock:
        return {
            "running":      train_state["running"],
            "epoch":        train_state["epoch"],
            "total_epochs": train_state["total_epochs"],
            "loss":         train_state["loss"],
            "recon_loss":   train_state["recon_loss"],
            "kl_loss":      train_state["kl_loss"],
            "done":         train_state["done"],
            "logs":         train_state["logs"][-50:],   # last 50 log lines
            "history":      train_state["history"],
            "error":        train_state["error"],
        }


# ── POST /generate ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    antigravity_score: float = 0.0   # -2 to +2
    element: str = "C"


@app.post("/generate")
def generate_crystal(req: GenerateRequest):
    """Generate one crystal structure conditioned on antigravity_score."""
    if _model is None or _scaler is None:
        raise HTTPException(400, "Model not trained yet. Train the model first.")

    _model.eval()
    params_sc = _model.generate(n=1, antigravity_score=req.antigravity_score)
    params    = _scaler.inverse_transform(params_sc.numpy())
    row       = params[0]

    # Build CIF
    cif_str = params_to_cif(row, element=req.element)

    # Encode CIF as base64 so the frontend can treat it as a downloadable blob
    cif_b64 = base64.b64encode(cif_str.encode()).decode()

    feature_names = ["num_atoms", "volume", "a", "b", "c", "alpha", "beta", "gamma"]
    params_dict   = {k: round(float(v), 4) for k, v in zip(feature_names, row)}

    return {
        "params":          params_dict,
        "antigravity_score": req.antigravity_score,
        "cif_base64":      cif_b64,
        "element":         req.element,
    }


# ── POST /validate ────────────────────────────────────────────────────────────
class ValidateRequest(BaseModel):
    cif_text: str


@app.post("/validate")
def validate_structure(req: ValidateRequest):
    """
    Parse a CIF string with pymatgen and return:
      - min_distance (Å)   — nearest-neighbour distance
      - density (g/cm³)
      - volume (Å³)
      - is_valid (bool)    — True if min_distance > 0.5 Å
    """
    from pymatgen.core import Structure

    try:
        struct = Structure.from_str(req.cif_text, fmt="cif")
    except Exception as e:
        raise HTTPException(400, f"Could not parse CIF: {e}")

    # Min interatomic distance
    dist_matrix  = struct.distance_matrix
    np.fill_diagonal(dist_matrix, np.inf)
    min_distance = float(dist_matrix.min())

    density = float(struct.density)
    volume  = float(struct.volume)
    is_valid = min_distance > 0.5    # atoms shouldn't overlap

    return {
        "min_distance": round(min_distance, 4),
        "density":      round(density,      4),
        "volume":       round(volume,        4),
        "num_atoms":    len(struct),
        "is_valid":     is_valid,
        "message":      "✅ Valid structure" if is_valid else "❌ Atoms too close — invalid structure",
    }


# ── GET /generate-multiple ────────────────────────────────────────────────────
@app.get("/generate-multiple")
def generate_multiple(
    n: int = 5,
    antigravity_score: float = 0.0,
    element: str = "C",
):
    """Generate `n` crystals at once and return their parameters."""
    if _model is None or _scaler is None:
        raise HTTPException(400, "Model not trained yet.")

    _model.eval()
    params_sc = _model.generate(n=n, antigravity_score=antigravity_score)
    params    = _scaler.inverse_transform(params_sc.numpy())

    feature_names = ["num_atoms", "volume", "a", "b", "c", "alpha", "beta", "gamma"]
    results = []
    for i, row in enumerate(params):
        p = {k: round(float(v), 4) for k, v in zip(feature_names, row)}
        p["index"] = i
        results.append(p)

    return {"samples": results, "antigravity_score": antigravity_score}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model_loaded": _model is not None}
