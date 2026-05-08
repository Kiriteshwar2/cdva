import React from "react";

export default function ComparisonMetrics({ primary, comparison }) {
  const formatValue = (val) => {
    if (val === null || val === undefined) return "—";
    if (typeof val === "number") {
      return val.toFixed(2);
    }
    return val;
  };

  const compareValues = (primary, comparison) => {
    if (!primary || !comparison) return "—";
    if (primary === comparison) return "match";
    return "diff";
  };

  return (
    <div className="comparison-metrics">
      <div className="metrics-panel">
        <h4>Primary Structure</h4>
        <div className="metrics-row">
          <div className="metric-item">
            <span className="metric-label">Formula</span>
            <span className="metric-value">{primary?.formula || "—"}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Atoms</span>
            <span className="metric-value">{primary?.atom_count || "—"}</span>
          </div>
        </div>
        <div className="metrics-row">
          <div className="metric-item">
            <span className="metric-label">Density</span>
            <span className="metric-value">
              {formatValue(primary?.density)} g/cm³
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Volume</span>
            <span className="metric-value">
              {formatValue(primary?.volume)} Å³
            </span>
          </div>
        </div>
        <div className="metrics-row">
          <div className="metric-item">
            <span className="metric-label">Space Group</span>
            <span className="metric-value">{primary?.space_group || "—"}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Status</span>
            <span
              className={`metric-badge ${
                primary?.is_valid ? "match" : "diff"
              }`}
            >
              {primary?.is_valid ? "Valid" : "Review"}
            </span>
          </div>
        </div>
      </div>

      {comparison && (
        <div className="metrics-panel">
          <h4>Comparison</h4>
          <div className="metrics-row">
            <div className="metric-item">
              <span className="metric-label">Formula</span>
              <span
                className={`metric-value ${
                  compareValues(primary?.formula, comparison?.formula) ===
                  "match"
                    ? ""
                    : "highlight"
                }`}
              >
                {comparison?.formula || "—"}
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Atoms</span>
              <span
                className={`metric-value ${
                  compareValues(primary?.atom_count, comparison?.atom_count) ===
                  "match"
                    ? ""
                    : "highlight"
                }`}
              >
                {comparison?.atom_count || "—"}
              </span>
            </div>
          </div>
          <div className="metrics-row">
            <div className="metric-item">
              <span className="metric-label">Density</span>
              <span
                className={`metric-value ${
                  compareValues(primary?.density, comparison?.density) ===
                  "match"
                    ? ""
                    : "highlight"
                }`}
              >
                {formatValue(comparison?.density)} g/cm³
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Volume</span>
              <span
                className={`metric-value ${
                  compareValues(primary?.volume, comparison?.volume) === "match"
                    ? ""
                    : "highlight"
                }`}
              >
                {formatValue(comparison?.volume)} Å³
              </span>
            </div>
          </div>
          <div className="metrics-row">
            <div className="metric-item">
              <span className="metric-label">Space Group</span>
              <span
                className={`metric-value ${
                  compareValues(
                    primary?.space_group,
                    comparison?.space_group
                  ) === "match"
                    ? ""
                    : "highlight"
                }`}
              >
                {comparison?.space_group || "—"}
              </span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Status</span>
              <span
                className={`metric-badge ${
                  comparison?.is_valid ? "match" : "diff"
                }`}
              >
                {comparison?.is_valid ? "Valid" : "Review"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
