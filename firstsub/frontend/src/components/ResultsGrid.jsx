import StructureCard from "./StructureCard";

export default function ResultsGrid({ structures }) {
  if (!structures.length) {
    return (
      <div className="results-empty">
        <p className="results-empty__eyebrow">Results</p>
        <h2 className="results-empty__title">No structures generated yet</h2>
        <p className="results-empty__text">
          Request one or more samples and the generated crystal cards will appear here with validation details, CIF exports, and 3D previews.
        </p>
      </div>
    );
  }

  return (
    <section className="dashboard-section">
      <div className="results-header">
        <div>
          <p className="dashboard-summary__kicker">Results</p>
          <h2 className="dashboard-summary__id" style={{ marginTop: 8, fontSize: "1.65rem" }}>Generated structures</h2>
        </div>
        <p className="results-count">{structures.length} returned from backend</p>
      </div>
      <div className="dashboard-results-grid">
        {structures.map((structure) => (
          <StructureCard key={structure.id} structure={structure} />
        ))}
      </div>
    </section>
  );
}
