# CDVAE Crystal Platform

Production-ready FastAPI + React platform for authenticated crystal generation, Mongo-backed experiment history, and checkpoint-driven CDVAE inference.

## Structure

- `backend/app/` is the canonical FastAPI backend.
- `backend/app/ml/` contains the CDVAE model, graph dataset code, and reusable ML utilities.
- `frontend/src/` contains the React application with the Three.js crystal landing page.
- `data/` contains dataset assets under `carbon_24/`, `mp_20/`, and `perov_5/`.
- `models/` is reserved for `.pt` checkpoints and weights.
- `scripts/` contains operational CLI workflows.

## Environment

Backend reads `backend/.env` locally and the same variable names in production:

```env
MONGO_URI=
JWT_SECRET=
MODEL_PATH=
ENV=production
```

Frontend reads:

```env
VITE_API_URL=
```

Use MongoDB Atlas in production by setting `MONGO_URI` to the Atlas connection string. The compose file provides local MongoDB for development only.

## Local Docker Runtime

```powershell
docker compose up --build
```

Frontend: `http://localhost:3000`
Backend: `http://localhost:8000`
Health: `http://localhost:8000/health`

## Supported API

- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/me`
- `GET /health`
- `GET /models`
- `POST /generate`
- `GET /history`
- `GET /generation/{id}`

Place a trained `.pt` checkpoint in `models/` or set `MODEL_PATH` to an explicit checkpoint path before using generation.
