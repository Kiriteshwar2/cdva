import CrystalViewer from "./CrystalViewer";

function downloadCif(structure) {
  const blob = new Blob([structure.cif_string], { type: "chemical/x-cif" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${structure.id}.cif`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function StructureCard({ structure }) {
  const { metadata } = structure;
  const isValid = Boolean(structure.valid);

  return (
    <article className="result-card">
      <div className="result-card__header">
        <div>
          <p className="result-card__eyebrow">Structure</p>
          <h3 className="result-card__title">{structure.id}</h3>
          <p className="result-card__formula">{metadata.formula || "Unknown formula"}</p>
        </div>
        <span className={`result-card__status ${isValid ? "result-card__status--valid" : "result-card__status--invalid"}`}>
          {isValid ? "Valid" : "Needs review"}
        </span>
      </div>

      <div className="result-card__body">
        <div className="result-card__stack">
          <div className="result-card__summary">
          <div className="result-card__stats">
            <div className="result-card__stat">
              <p className="result-card__stat-label">Atoms</p>
              <p className="result-card__stat-value">{metadata.num_atoms}</p>
            </div>
            <div className="result-card__stat">
              <p className="result-card__stat-label">Volume</p>
              <p className="result-card__stat-value">{metadata.volume.toFixed(2)}</p>
            </div>
            <div className="result-card__stat">
              <p className="result-card__stat-label">Min Distance</p>
              <p className="result-card__stat-value">
                {Number.isFinite(metadata.minimum_pair_distance)
                  ? metadata.minimum_pair_distance.toFixed(2)
                  : "n/a"}
              </p>
            </div>
          </div>

          <div className="chip-list">
            {structure.atom_types.map((atomType, index) => (
              <span key={`${structure.id}-${atomType}-${index}`} className="result-card__chip">
                {atomType}
              </span>
            ))}
          </div>
          </div>

          <div className="result-card__viewer">
            <div className="result-card__viewer-title">
              <p className="dashboard-summary__kicker" style={{ letterSpacing: "0.2em" }}>3D Viewer</p>
            </div>
            <CrystalViewer cifString={structure.cif_string} structureId={structure.id} bonds={structure.bonds || []} />
          </div>

          <details className="result-card__cif">
            <summary>CIF export</summary>
            <div style={{ padding: "0 16px 16px" }}>
              <div className="result-card__viewer-title">
                <p className="dashboard-summary__kicker" style={{ letterSpacing: "0.2em" }}>CIF Preview</p>
                <button
                  type="button"
                  onClick={() => downloadCif(structure)}
                  className="btn-outline"
                >
                  Download CIF
                </button>
              </div>
              <pre>{structure.cif_string}</pre>
            </div>
          </details>
          </div>
      </div>
    </article>
  );
}
