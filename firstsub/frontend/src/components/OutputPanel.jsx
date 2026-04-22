// OutputPanel.jsx — Show generated crystal parameters + CIF download
export default function OutputPanel({ data }) {
  if (!data) return (
    <div className="card" style={{
      display:"flex", alignItems:"center", justifyContent:"center",
      minHeight:130, color:"#6b7280", fontSize:"0.85rem", textAlign:"center",
    }}>
      No crystal generated yet.<br/>Use the Generate panel.
    </div>
  );

  const { params, cif_base64, antigravity_score, element } = data;

  const handleDownload = () => {
    const cif  = atob(cif_base64);
    const blob = new Blob([cif], { type:"text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `crystal_${element}_score${antigravity_score}.cif`;
    a.click(); URL.revokeObjectURL(url);
  };

  const rows = [
    ["Atoms",  params.num_atoms, ""],
    ["Volume", params.volume,    "Å³"],
    ["a",      params.a,         "Å"],
    ["b",      params.b,         "Å"],
    ["c",      params.c,         "Å"],
    ["α",      params.alpha,     "°"],
    ["β",      params.beta,      "°"],
    ["γ",      params.gamma,     "°"],
  ];

  return (
    <div className="card" style={{ display:"flex", flexDirection:"column", gap:"0.75rem" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:4 }}>
        <h2 style={{ color:"var(--accent)", fontWeight:700, fontSize:"1.05rem" }}>🧬 Generated Crystal</h2>
        <span style={{ fontSize:"0.74rem", color:"#9ca3af" }}>
          {element} · score <span style={{ color:"var(--purple)", fontWeight:700 }}>{antigravity_score}</span>
        </span>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"0.5rem" }}>
        {rows.map(([label, val, unit]) => (
          <div className="metric-card" key={label}>
            <div className="metric-label">{label}</div>
            <div className="metric-value">{typeof val === "number" ? val.toFixed(3) : val}</div>
            {unit && <div className="metric-unit">{unit}</div>}
          </div>
        ))}
      </div>

      {cif_base64 && (
        <button className="btn-secondary" onClick={handleDownload}>
          ⬇ Download CIF
        </button>
      )}
    </div>
  );
}
