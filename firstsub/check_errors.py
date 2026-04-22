import json, re

ansi = re.compile(r'\x1b\[[0-9;]*m')

with open("crystal_vae_test_carbon24.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code_cells):
    src = "".join(c.get("source", []))[:60].replace("\n", " ")
    outs = c.get("outputs", [])
    err_outs = [o for o in outs if o.get("output_type") == "error"]
    stdout = [o for o in outs if o.get("output_type") == "stream" and o.get("name") == "stdout"]
    if err_outs:
        print(f"\n=== ERROR in Cell {i+1} ===")
        print(f"  src: {src}")
        for o in err_outs:
            print(f"  {o['ename']}: {o['evalue']}")
            for t in o.get("traceback", [])[-6:]:
                print("   ", ansi.sub("", t))
    else:
        out_text = "".join("".join(o.get("text", [])) for o in stdout)[:120].replace("\n", " | ")
        print(f"Cell {i+1}: OK | {out_text or '(no stdout)'}")
print("\nDone.")
