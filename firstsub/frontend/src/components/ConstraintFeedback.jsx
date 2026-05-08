// ConstraintFeedback.jsx — Show constraint satisfaction status
export default function ConstraintFeedback({ violations, generationStatus }) {
  if (!violations || violations.length === 0) {
    return (
      <div className="constraint-feedback">
        <h4 className="constraint-feedback__title">✓ Constraint Status</h4>
        <div className="constraint-item success">
          <div className="constraint-item__icon">✓</div>
          <div className="constraint-item__text">
            All constraints satisfied
          </div>
        </div>
      </div>
    );
  }

  const parseViolation = (violation) => {
    if (violation.includes("Atom count:")) {
      return { type: "warning", icon: "⚠", text: violation };
    } else if (violation.includes("too")) {
      return { type: "warning", icon: "⚠", text: violation };
    } else if (violation.includes("failed") || violation.includes("Fallback")) {
      return { type: "error", icon: "✕", text: violation };
    } else {
      return { type: "warning", icon: "◐", text: violation };
    }
  };

  return (
    <div className="constraint-feedback">
      <h4 className="constraint-feedback__title">
        {generationStatus === "partial_success" ? "⚡ Partial Success" : "ℹ Constraint Status"}
      </h4>
      {generationStatus === "partial_success" && (
        <p style={{ fontSize: "0.9rem", color: "var(--muted)", margin: "0 0 12px 0" }}>
          We adapted your constraints and generated the closest possible crystal structure.
        </p>
      )}
      {violations.map((violation, idx) => {
        const parsed = parseViolation(violation);
        return (
          <div key={idx} className={`constraint-item ${parsed.type}`}>
            <div className="constraint-item__icon">{parsed.icon}</div>
            <div className="constraint-item__text">{parsed.text}</div>
          </div>
        );
      })}
      <div style={{ marginTop: "12px", paddingTop: "12px", borderTop: "1px solid var(--panel-border)" }}>
        <p style={{ fontSize: "0.85rem", color: "var(--muted)", margin: "0" }}>
          <strong>💡 Suggestions:</strong>
        </p>
        <ul style={{ fontSize: "0.85rem", color: "var(--muted)", margin: "8px 0 0 20px", paddingLeft: 0 }}>
          <li>Try adjusting constraint ranges to be less strict</li>
          <li>Increase maximum attempts for better candidates</li>
          <li>Relax interatomic distance if too restrictive</li>
        </ul>
      </div>
    </div>
  );
}
