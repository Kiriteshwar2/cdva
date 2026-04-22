import json

with open("crystal_generator_vae.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total code cells: {len(code_cells)}")
for i, c in enumerate(code_cells):
    src = "".join(c.get("source", []))[:80].replace("\n", " ")
    outputs = c.get("outputs", [])
    err = any(o.get("output_type") == "error" for o in outputs)
    print(f"  Cell {i+1}: src={repr(src)} | outputs={len(outputs)} | error={err}")
