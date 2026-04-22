import json, re

ansi = re.compile(r'\x1b\[[0-9;]*m')

with open("crystal_vae_test_carbon24.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
# Print stdout from metrics cell (cell 7, index 6) and generation cell
for i in [6, 7, 8]:
    c = code_cells[i]
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print(f"=== Cell {i+1} ===")
            print("".join(o.get("text", [])))

# Check CIF file saved
import os
for f in ["generated_carbon24.cif"]:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"CIF file '{f}': exists={exists}, size={size} bytes")
