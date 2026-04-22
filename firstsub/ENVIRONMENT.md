# Environment Setup

This project is intended to run on Python 3.11 or Python 3.12.

## 1. Create a clean virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `py -3.12` is not available, install Python 3.12 first and then rerun the command above. Python 3.10 is not recommended for this project because recent `mp-api` and `emmet-core` releases depend on newer typing behavior and modern `pydantic`.

## 2. Install dependencies

CPU-only:

```powershell
pip install -r requirements.txt
```

CUDA:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If you preinstall a CUDA-enabled `torch`, `pip install -r requirements.txt` will keep the rest of the stack aligned.

## 3. Verify the environment

```powershell
python -c "from mp_api.client import MPRester; import torch; import pymatgen; import e3nn; import fastapi; print('environment_ok')"
```

## 4. Configure Materials Project access

Set your API key before building the dataset:

```powershell
$env:MP_API_KEY = "your_materials_project_api_key"
```

Persist it for future terminals (recommended):

```powershell
setx MP_API_KEY "your_materials_project_api_key"
```

After `setx`, restart the terminal so the variable is available in new sessions.

## 5. Build a small dataset sanity check

```powershell
python -m data.mp_dataset --max-materials 1000 --max-elements 3 --max-atoms 50 --chunk-size 250
```

This should create:

- `processed_structures.jsonl`
- `species_vocab.json`

inside the computed cache directory under `cache/mp/`.
