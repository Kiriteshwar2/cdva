// CIFDownloadPanel.jsx — CIF file download and viewing
export default function CIFDownloadPanel({ cifString, formula = "crystal", onLog }) {
  const handleDownloadCIF = () => {
    const blob = new Blob([cifString], { type: "text/plain; charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${formula.replace(/\s+/g, "_")}.cif`;
    link.click();
    URL.revokeObjectURL(url);
    onLog?.(`Downloaded CIF: ${formula}.cif`);
  };

  const handleCopyCIF = () => {
    navigator.clipboard.writeText(cifString).then(() => {
      onLog?.("CIF data copied to clipboard");
    });
  };

  const handleExportJSON = () => {
    const payload = JSON.stringify({ formula, cif: cifString }, null, 2);
    const blob = new Blob([payload], { type: "application/json; charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${formula.replace(/\s+/g, "_")}.json`;
    link.click();
    URL.revokeObjectURL(url);
    onLog?.(`Exported JSON: ${formula}.json`);
  };

  const cifLines = cifString ? cifString.split("\n").length : 0;

  return (
    <div className="studio-card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div className="panel-heading">
        <h3 style={{ margin: 0 }}>Crystal Information File (CIF)</h3>
      </div>

      {/* File Info */}
      <div style={{
        padding: "12px",
        background: "var(--bg-soft)",
        borderRadius: "8px",
        marginBottom: "12px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <p style={{ margin: "0 0 4px 0", fontSize: "0.85rem", color: "var(--muted)" }}>
            CIF Format
          </p>
          <p style={{ margin: 0, fontSize: "0.9rem", fontWeight: 600, color: "var(--text)" }}>
            {cifLines} lines • {(cifString.length / 1024).toFixed(1)} KB
          </p>
        </div>
        <div style={{ fontSize: "2rem" }}>📄</div>
      </div>

      {/* CIF Preview */}
      <details style={{
        marginBottom: "12px",
        border: "1px solid var(--panel-border)",
        borderRadius: "8px",
        overflow: "hidden",
      }}>
        <summary style={{
          padding: "12px",
          background: "var(--bg-soft)",
          cursor: "pointer",
          fontWeight: 600,
          fontSize: "0.9rem",
          color: "var(--text)",
          userSelect: "none",
        }}>
          Preview CIF Structure
        </summary>
        <pre style={{
          margin: 0,
          padding: "12px",
          background: "var(--bg)",
          color: "var(--text)",
          fontSize: "0.75rem",
          lineHeight: 1.4,
          maxHeight: "200px",
          overflow: "auto",
          fontFamily: "'Courier New', monospace",
          whiteSpace: "pre-wrap",
          wordWrap: "break-word",
        }}>
          {cifString}
        </pre>
      </details>

      {/* Action Buttons */}
      <div style={{ display: "grid", gap: "8px", gridTemplateColumns: "1fr 1fr" }}>
        <button
          onClick={handleDownloadCIF}
          className="btn-primary"
          style={{
            flex: 1,
            padding: "10px 12px",
            fontSize: "0.9rem",
            justifyContent: "center",
          }}
        >
          ⬇ Download CIF
        </button>
        <button
          onClick={handleCopyCIF}
          className="btn-outline"
          style={{
            flex: 1,
            padding: "10px 12px",
            fontSize: "0.9rem",
            justifyContent: "center",
          }}
        >
          ⧉ Copy Data
        </button>
        <button onClick={handleExportJSON} className="btn-outline" style={{ gridColumn: "span 2", padding: "10px 12px", fontSize: "0.9rem", justifyContent: "center" }}>
          ⇲ Export JSON
        </button>
      </div>

      {/* Info */}
      <p style={{
        fontSize: "0.8rem",
        color: "var(--muted)",
        marginTop: "12px",
        marginBottom: 0,
        fontStyle: "italic",
      }}>
        CIF format is compatible with VASP, LAMMPS, ASE, and other materials simulation tools.
      </p>
    </div>
  );
}
