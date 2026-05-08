import React from "react";

export default function SharedViewerToolbar({
  onReset,
  onScreenshot,
  onToggleBonds,
  onToggleAtoms,
  onToggleCell,
  onAutoRotate,
  onFullscreen,
  showBonds = true,
  showAtoms = true,
  showCell = true,
  autoRotate = false,
}) {
  return (
    <div className="shared-viewer-toolbar">
      <button
        title="Reset view"
        onClick={onReset}
        className="toolbar-button"
      >
        ↻
      </button>

      <button
        title="Screenshot"
        onClick={onScreenshot}
        className="toolbar-button"
      >
        📷
      </button>

      <div className="toolbar-divider"></div>

      <button
        title="Toggle atoms"
        onClick={onToggleAtoms}
        className={`toolbar-button ${showAtoms ? "active" : ""}`}
      >
        ●
      </button>

      <button
        title="Toggle bonds"
        onClick={onToggleBonds}
        className={`toolbar-button ${showBonds ? "active" : ""}`}
      >
        −
      </button>

      <button
        title="Toggle unit cell"
        onClick={onToggleCell}
        className={`toolbar-button ${showCell ? "active" : ""}`}
      >
        ⊡
      </button>

      <div className="toolbar-divider"></div>

      <button
        title="Auto rotate"
        onClick={onAutoRotate}
        className={`toolbar-button ${autoRotate ? "active" : ""}`}
      >
        ⟳
      </button>

      <button
        title="Fullscreen"
        onClick={onFullscreen}
        className="toolbar-button"
      >
        ⛶
      </button>
    </div>
  );
}
