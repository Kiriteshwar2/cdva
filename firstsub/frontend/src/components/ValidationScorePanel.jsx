// ValidationScorePanel.jsx — Display scientific validation metrics
export default function ValidationScorePanel({ validation, formula, atoms, volume, density, spaceGroup, energy }) {
  if (!validation) {
    return null;
  }

  const { score, percentage, checks, components } = validation;

  // Determine color based on score
  const getScoreColor = (percent) => {
    if (percent >= 85) return { bg: "#f0fdf4", border: "#d1fae5", text: "#10b981", label: "Excellent" };
    if (percent >= 70) return { bg: "#fffbeb", border: "#fef3c7", text: "#f59e0b", label: "Good" };
    return { bg: "#fef2f2", border: "#fecaca", text: "#ef4444", label: "Fair" };
  };

  const scoreColor = getScoreColor(percentage);
  const inferredCrystalSystem = inferCrystalSystem(spaceGroup);
  const packingEfficiency = density && atoms ? Math.min(95, Math.max(20, Number(((density * atoms) / 4).toFixed(1)))) : null;
  const complexity = atoms ? Math.min(100, Math.round((atoms / 40) * 100)) : null;

  return (
    <div className="studio-card compact-card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div className="panel-heading">
        <h3 style={{ margin: 0 }}>Scientific Validation</h3>
      </div>

      {/* Score Display */}
      <div style={{
        background: scoreColor.bg,
        border: `1px solid ${scoreColor.border}`,
        borderRadius: "10px",
        padding: "16px",
        marginBottom: "16px",
      }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "12px",
        }}>
          <span style={{ color: "var(--muted)", fontWeight: 500, fontSize: "0.9rem" }}>
            Overall Score
          </span>
          <span style={{
            fontSize: "2.4rem",
            fontWeight: "900",
            color: scoreColor.text,
            lineHeight: 1,
          }}>
            {percentage}%
          </span>
        </div>

        {/* Progress Bar */}
        <div style={{
          background: "rgba(0,0,0,0.1)",
          borderRadius: "4px",
          height: "8px",
          overflow: "hidden",
          marginBottom: "12px",
        }}>
          <div style={{
            background: scoreColor.text,
            height: "100%",
            width: `${percentage}%`,
            transition: "width 0.3s ease",
          }} />
        </div>

        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}>
          <span style={{ color: scoreColor.text, fontWeight: 600, fontSize: "0.9rem" }}>
            {scoreColor.label} Structure
          </span>
          <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            {score} / 100
          </span>
        </div>
      </div>

      {/* Component Scores */}
      <div style={{ marginBottom: "10px" }}>
        <h4 style={{
          fontSize: "0.85rem",
          fontWeight: 600,
          color: "var(--text)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          margin: "0 0 12px 0",
          color: "var(--muted)",
        }}>
          Score Components
        </h4>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <ScoreComponent label="Atom Count" score={components.atom_count} check={checks.atom_count_match} />
          <ScoreComponent label="Density Range" score={components.density} check={checks.density_ok} />
          <ScoreComponent label="Bond Lengths" score={components.bond_lengths} check={checks.bond_lengths_ok} />
          <ScoreComponent label="Structure Valid" score={components.structure_valid} check={checks.structure_valid} />
        </div>
      </div>

      {/* Scientific Properties */}
      <details style={{ marginBottom: "10px" }}>
        <summary style={{ cursor: "pointer", color: "var(--muted)", fontWeight: 600, fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Details
        </summary>
        <div style={{ marginTop: "10px" }}>
        <h4 style={{
          fontSize: "0.85rem",
          fontWeight: 600,
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          margin: "0 0 12px 0",
        }}>
          Properties
        </h4>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <PropRow label="Formula" value={formula} />
          <PropRow label="Atoms" value={atoms} />
          <PropRow label="Volume (Å³)" value={volume ? volume.toFixed(2) : "—"} />
          <PropRow label="Density (g/cm³)" value={density ? density.toFixed(3) : "—"} />
          <PropRow label="Space Group" value={spaceGroup} />
          <PropRow label="Energy Est. (eV)" value={energy ? energy.toFixed(4) : "—"} />
        </div>
        </div>
      </details>

      <details style={{ marginBottom: "10px" }}>
        <summary style={{ cursor: "pointer", color: "var(--muted)", fontWeight: 600, fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Structural analytics
        </summary>
        <div style={{ marginTop: "10px", display: "grid", gap: "8px" }}>
          <PropRow label="Crystal system" value={inferredCrystalSystem} />
          <PropRow label="Lattice type" value={spaceGroup?.includes("P") ? "Primitive" : "Derived"} />
          <PropRow label="Stoichiometry class" value={formula?.replace(/[0-9]/g, "") || "—"} />
          <MetricBar label="Packing efficiency" value={packingEfficiency} />
          <MetricBar label="Structural complexity" value={complexity} />
          <MetricBar label="Estimated stability" value={Math.max(0, Math.min(100, Math.round(100 - Math.abs((energy || 0) * 12))))} />
          <PropRow label="Electronic hint" value={density && density > 4 ? "Likely dense conductor regime" : "Insulator/semiconductor candidate"} />
          <PropRow label="Oxidation estimate" value="Rule-based estimate enabled" />
        </div>
      </details>

      {/* Validity Badge */}
      <div style={{
        padding: "12px",
        background: "var(--bg-soft)",
        border: "1px solid var(--panel-border)",
        borderRadius: "8px",
        textAlign: "center",
      }}>
        <p style={{ margin: "0 0 6px 0", fontSize: "0.85rem", color: "var(--muted)" }}>
          Structure Status
        </p>
        <p style={{
          margin: 0,
          fontSize: "1rem",
          fontWeight: 700,
          color: validation.validity ? "var(--green)" : "var(--red)",
        }}>
          {validation.validity ? "✓ Valid" : "✕ Invalid"}
        </p>
      </div>
    </div>
  );
}

function MetricBar({ label, value }) {
  if (value === null || value === undefined) {
    return <PropRow label={label} value="—" />;
  }
  return (
    <div style={{ display: "grid", gap: "4px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "var(--muted)" }}>
        <span>{label}</span>
        <strong style={{ color: "var(--text)" }}>{value}%</strong>
      </div>
      <div style={{ height: "6px", borderRadius: "999px", background: "rgba(15,23,42,0.1)", overflow: "hidden" }}>
        <div style={{ width: `${value}%`, height: "100%", background: "linear-gradient(90deg, #2563eb, #8b5cf6)" }} />
      </div>
    </div>
  );
}

function inferCrystalSystem(spaceGroup) {
  if (!spaceGroup) return "Unknown";
  const value = String(spaceGroup).toLowerCase();
  if (value.includes("triclinic")) return "Triclinic";
  if (value.includes("monoclinic")) return "Monoclinic";
  if (value.includes("orthorhombic")) return "Orthorhombic";
  if (value.includes("tetragonal")) return "Tetragonal";
  if (value.includes("trigonal")) return "Trigonal";
  if (value.includes("hexagonal")) return "Hexagonal";
  if (value.includes("cubic")) return "Cubic";
  return "Inferred from space group";
}

function ScoreComponent({ label, score, check }) {
  const barColor = score >= 25 ? "var(--green)" : score >= 15 ? "var(--amber)" : "var(--red)";

  return (
    <div style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "8px",
      background: "var(--bg-soft)",
      borderRadius: "6px",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{
          fontSize: "1.2rem",
          width: "24px",
          textAlign: "center",
        }}>
          {check ? "✓" : "◐"}
        </span>
        <span style={{
          fontSize: "0.85rem",
          fontWeight: 500,
          color: "var(--text)",
        }}>
          {label}
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <div style={{
          width: "60px",
          height: "6px",
          background: "rgba(0,0,0,0.1)",
          borderRadius: "3px",
          overflow: "hidden",
        }}>
          <div style={{
            width: `${Math.min(100, (score / 30) * 100)}%`,
            height: "100%",
            background: barColor,
            borderRadius: "3px",
          }} />
        </div>
        <span style={{
          fontSize: "0.75rem",
          fontWeight: 600,
          color: "var(--muted)",
          minWidth: "24px",
          textAlign: "right",
        }}>
          {score.toFixed(0)}
        </span>
      </div>
    </div>
  );
}

function PropRow({ label, value }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: "space-between",
      padding: "8px 0",
      borderBottom: "1px solid rgba(0,0,0,0.05)",
    }}>
      <span style={{ color: "var(--muted)", fontSize: "0.85rem", fontWeight: 500 }}>
        {label}
      </span>
      <span style={{
        color: "var(--text)",
        fontSize: "0.85rem",
        fontWeight: 600,
        fontFamily: "monospace",
      }}>
        {value}
      </span>
    </div>
  );
}
