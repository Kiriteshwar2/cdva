import CIFDownloadPanel from "./CIFDownloadPanel";
import ValidationScorePanel from "./ValidationScorePanel";

export default function ValidationExportRail({ generation, onLog }) {
  if (!generation?.metadata) return null;

  const { metadata, output_cif, structure } = generation;

  return (
    <section className="validation-stack">
      <div className="validation-summary-card">
        <div>
          <span className="validation-label">Structure Status</span>
          <h3>{metadata.validity ? "Valid Structure" : "Needs Review"}</h3>
        </div>

        <div className={`validation-pill ${metadata.validity ? "good" : "warn"}`}>
          {metadata.validity ? "VALID" : "REVIEW"}
        </div>
      </div>

      <div className="validation-metrics-grid">
        <div className="metric-card">
          <span>Atoms</span>
          <strong>{metadata.atoms_count || "—"}</strong>
        </div>

        <div className="metric-card">
          <span>Density</span>
          <strong>
            {metadata.density?.toFixed?.(2) || "—"}
          </strong>
        </div>

        <div className="metric-card">
          <span>Volume</span>
          <strong>
            {metadata.volume?.toFixed?.(2) || "—"}
          </strong>
        </div>

        <div className="metric-card">
          <span>Space Group</span>
          <strong>{metadata.space_group || "—"}</strong>
        </div>
      </div>

      <ValidationScorePanel
        validation={metadata.validation}
        formula={structure?.formula}
        atoms={metadata.atoms_count}
        volume={metadata.volume}
        density={metadata.density}
        spaceGroup={metadata.space_group}
        energy={metadata.energy_estimate}
      />

      <CIFDownloadPanel
        cifString={output_cif || ""}
        formula={structure?.formula || "Crystal"}
        onLog={onLog}
      />
    </section>
  );
}