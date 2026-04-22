"""Add missing 'id' fields to notebook cells (required by nbformat >= 4.5)."""
import json, uuid

with open("crystal_generator_vae.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

fixed = 0
for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = uuid.uuid4().hex[:8]
        fixed += 1

with open("crystal_generator_vae.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added id to {fixed} cells. Notebook updated.")
