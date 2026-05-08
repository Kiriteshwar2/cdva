import React, { useState, useEffect } from "react";
import StructureViewer3D from "./StructureViewer3D";

export default function ComparisonViewer({ generation, onSelectFromHistory }) {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(!!generation);
  }, [generation]);

  if (!isLoaded) {
    return (
      <div className={`comparison-viewer`}>
        <div className="comparison-empty">
          <div className="comparison-empty__icon">⊕</div>
          <h3>Compare Structure</h3>
          <p>Load another crystal to compare</p>
          <button
            onClick={onSelectFromHistory}
            className="secondary-button"
          >
            Select from history
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`comparison-viewer loaded`}>
      <div className="structure-viewer">
        <div className="structure-viewer__header">
          <div>
            <h2>Comparison</h2>
          </div>
        </div>
        <div className="structure-viewer__content">
          {generation?.cif_data ? (
            <StructureViewer3D
              cif_data={generation.cif_data}
              onScreenshot={() => {}}
            />
          ) : (
            <div className="structure-viewer__empty">
              <div className="structure-viewer__icon">◑</div>
              <h3>No structure data</h3>
              <p>CIF data unavailable for comparison</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
