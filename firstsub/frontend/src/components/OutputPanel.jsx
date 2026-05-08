// OutputPanel.jsx — Show generated crystal parameters + CIF download with constraint feedback
import ConstraintFeedback from "./ConstraintFeedback.jsx";

export default function OutputPanel({ data, onLog }) {
  if (!data || !data.metadata) return (
    <div className="studio-card" style={{ display: "flex", flexDirection: "column" }}>
      <div className="panel-heading">
        <h3>Generated Properties</h3>
      </div>
      <div style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        minHeight: 180, color: "var(--muted)", fontSize: "0.82rem", textAlign: "center",
        flexDirection: "column", gap: 8,
      }}>
        <span style={{ fontSize: "2rem" }}>⬡</span>
        <span>No crystal generated yet.</span>
        <span style={{ fontSize: "0.72rem" }}>Adjust parameters and generate to see results.</span>
      </div>
    </div>
  );

  const { metadata, cif_string, structure } = data;
  
  // Extract values from metadata
  const formula = structure?.formula || "—";
  const numAtoms = metadata.atoms_count || 0;
  const volume = metadata.volume || 0;
  const density = metadata.density || 0;
  const spaceGroup = metadata.space_group || "—";
  const energy = metadata.energy_estimate || 0;
  const lattice = metadata.lattice || {};
  const violations = metadata.constraint_violations || [];
  const generationStatus = metadata.generation_status || "success";
  const attempsUsed = metadata.attempts_used || 0;
  const totalEvaluated = metadata.total_candidates_evaluated || 0;
  
  const cifText = cif_string || "";

  const handleDownload = () => {
    const blob = new Blob([cifText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `crystal_${formula.replace(/\s+/g, "_")}.cif`;
    a.click();
    URL.revokeObjectURL(url);
    onLog?.(`Downloaded CIF: crystal_${formula.replace(/\s+/g, "_")}.cif`);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(cifText).then(() => onLog?.("CIF data copied to clipboard"));
  };

  const fmt = (v, d = 2) => (v != null && !isNaN(v)) ? Number(v).toFixed(d) : "—";

  // Determine status badge and color
  const isFullSuccess = !violations || violations.length === 0;
  const statusBadgeClass = isFullSuccess ? "badge-success" : "badge-warning";
  const statusText = isFullSuccess ? "✓ Perfect Match" : "⚡ Adapted Result";

  return (
    <div className="studio-card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div className="panel-heading">
        <div>
          <h3 style={{ margin: "0 0 6px 0" }}>Generated Properties</h3>
          <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--muted)" }}>
            Evaluated {totalEvaluated} candidates from {attempsUsed} attempts
          </p>
        </div>
        <span className={statusBadgeClass}>{statusText}</span>
      </div>

      {/* Status message */}
      {generationStatus === "partial_success" && (
        <div style={{
          padding: "12px",
          background: "#fffbeb",
          border: "1px solid #fef3c7",
          borderRadius: "8px",
          marginBottom: "16px",
          color: "#78350f",
          fontSize: "0.9rem",
          lineHeight: "1.5"
        }}>
          <strong>📌 Your exact constraints could not be fully satisfied.</strong> We generated the closest possible crystal structure based on your inputs. Review the constraint status below.
        </div>
      )}

      {/* Property rows */}
      <div style={{ flex: 1, marginBottom: "16px" }}>

        {/* Formula + atoms */}
        <div className="prop-row">
          <span className="prop-key">Formula</span>
          <span className="prop-value" style={{ fontFamily: "monospace", letterSpacing: 1 }}>{formula}</span>
        </div>
        <div className="prop-row">
          <span className="prop-key">Number of Atoms</span>
          <span className="prop-value">{numAtoms}</span>
        </div>
        <div className="prop-row">
          <span className="prop-key">Volume (Å³)</span>
          <span className="prop-value">{fmt(volume, 3)}</span>
        </div>
        <div className="prop-row">
          <span className="prop-key">Density (g/cm³)</span>
          <span className="prop-value">{fmt(density, 3)}</span>
        </div>
        <div className="prop-row">
          <span className="prop-key">Energy Estimate (eV)</span>
          <span className="prop-value">{fmt(energy, 4)}</span>
        </div>

        {/* Lattice section header */}
        <div style={{
          fontSize: "0.7rem", fontWeight: 700, color: "var(--muted)",
          textTransform: "uppercase", letterSpacing: "0.05em",
          padding: "10px 0 4px",
        }}>
          Lattice Parameters (Å)
        </div>
        {[["a", lattice.a], ["b", lattice.b], ["c", lattice.c]].map(([k, v]) => (
          <div className="prop-row" key={k} style={{ paddingLeft: 8 }}>
            <span className="prop-key">{k}</span>
            <span className="prop-value">{fmt(v, 3)}</span>
          </div>
        ))}

        {/* Angles section header */}
        <div style={{
          fontSize: "0.7rem", fontWeight: 700, color: "var(--muted)",
          textTransform: "uppercase", letterSpacing: "0.05em",
          padding: "10px 0 4px",
        }}>
          Angles (°)
        </div>
        {[["α", lattice.alpha], ["β", lattice.beta], ["γ", lattice.gamma]].map(([k, v]) => (
          <div className="prop-row" key={k} style={{ paddingLeft: 8 }}>
            <span className="prop-key">{k}</span>
            <span className="prop-value">{fmt(v, 2)}</span>
          </div>
        ))}

        <div className="prop-row">
          <span className="prop-key">Space Group</span>
          <span className="prop-value">{spaceGroup}</span>
        </div>
        <div className="prop-row">
          <span className="prop-key">Min Interatomic Distance (Å)</span>
          <span className="prop-value">{fmt(metadata.min_interatomic_distance, 3)}</span>
        </div>
      </div>

      {/* Constraint Feedback Panel */}
      <ConstraintFeedback violations={violations} generationStatus={generationStatus} />

      {/* CIF Details */}
      {cifText && (
        <details className="cif-panel" style={{ marginTop: "16px" }}>
          <summary>View CIF Structure File</summary>
          <pre>{cifText}</pre>
        </details>
      )}

      {/* Action buttons */}
      {cifText && (
        <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
          <button className="btn-primary" onClick={handleDownload}
            style={{ flex: 1, justifyContent: "center" }}>
            ⬇ Download CIF
          </button>
          <button className="btn-outline" onClick={handleCopy}
            style={{ flex: 1, justifyContent: "center" }}>
            ⧉ Copy Data
          </button>
        </div>
      )}
    </div>
  );
}
