// StructureViewer3D.jsx — Production-safe 3D crystal visualization with 3Dmol.js
import { useEffect, useRef, useState } from "react";

export default function StructureViewer3D({ cifString, formula = "—" }) {
  const viewerContainer = useRef(null);
  const viewerShell = useRef(null);
  const viewer = useRef(null);
  const [viewMode, setViewMode] = useState("ball_stick");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAtoms, setShowAtoms] = useState(true);
  const [showBonds, setShowBonds] = useState(true);
  const [showCell, setShowCell] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const activeCif = cifString;

  // Clean CIF by removing comments
  const cleanCif = (cif) => {
    return cif
      .split("\n")
      .filter(line => !line.trim().startsWith("#"))
      .join("\n");
  };
  // Validate CIF format
  const isValidCif = (cif) => {
    if (!cif || typeof cif !== "string") return false;

    const cleaned = cif
        .split("\n")
        .map(line => line.trim())
        .filter(line => line && !line.startsWith("#"));

    return cleaned.some(line => line.startsWith("data_"));
  };

  // Load and initialize 3Dmol.js
  useEffect(() => {
    if (!viewerContainer.current) {
      return;
    }

    let mounted = true;

    const initializeViewer = async () => {
      try {
        setLoading(true);
        setError("");
        // Load 3Dmol.js if not already loaded
        if (!window.$3Dmol) {
          const script = document.createElement("script");
          script.src = "https://3Dmol.csb.pitt.edu/build/3Dmol-min.js";
          script.async = true;

          await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = () => reject(new Error("Failed to load 3Dmol.js library"));
            document.head.appendChild(script);
          });

        }

        if (!mounted) return;

        // Create viewer instance
        if (!viewerContainer.current) {
          return;
        }

        const viewer3d = window.$3Dmol.createViewer(viewerContainer.current, {
          backgroundColor: "#f8fafc",
          antialias: false, // ✅ Disabled for performance
          disableFog: true, // ✅ Better rendering
        });
        viewer.current = viewer3d;
        viewer3d.spin(false); // Disable auto-spin for performance

        // Render if CIF is available
        if (isValidCif(activeCif)) {
          renderStructure(viewer3d, activeCif, viewMode);
        } else {
          setLoading(false);
        }
      } catch (err) {
        if (mounted) {
          setError(err.message);
          setLoading(false);
        }
      }
    };

    initializeViewer();

    return () => {
      mounted = false;
    };
  }, []);

  // Load CIF - runs once when activeCif changes
  useEffect(() => {
    if (!viewer.current || !isValidCif(activeCif)) {
      return;
    }

    renderStructure(viewer.current, activeCif, viewMode);
  }, [activeCif]);

  // Update style only when viewMode changes - ✅ No re-render
  useEffect(() => {
    if (!viewer.current || !isValidCif(activeCif)) return;
    const style = getStyleForViewMode(viewMode, showAtoms, showBonds);
    viewer.current.setStyle({}, style);
    if (showCell) {
      viewer.current.addUnitCell();
    }
    viewer.current.render();
  }, [viewMode, showAtoms, showBonds, showCell]);

  useEffect(() => {
    if (!viewer.current) return;
    viewer.current.spin(Boolean(autoRotate));
    viewer.current.render();
  }, [autoRotate]);

  useEffect(() => {
    const handleResize = () => {
      if (viewer.current) {
        // Resize canvas to fit container
        viewer.current.resize();
        
        // Re-fit structure to visible area
        viewer.current.center();
        viewer.current.zoomTo();
        
        // Render updated view
        viewer.current.render();
      }
    };

    // Add resize listener with slight debounce
    let resizeTimer;
    const debouncedResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(handleResize, 150);
    };

    window.addEventListener("resize", debouncedResize);

    return () => {
      window.removeEventListener("resize", debouncedResize);
      clearTimeout(resizeTimer);
    };
  }, []);

  // Render the 3D structure with optimized performance
  const renderStructure = (viewerInstance, cif, mode) => {
    try {
      viewerInstance.clear();
      
      // Add CIF model
      const cleaned = cleanCif(cif);
      const model = viewerInstance.addModel(cleaned, "cif");
      
      // Apply styling
      const style = getStyleForViewMode(mode, showAtoms, showBonds);
      model.setStyle({}, style);
      
      // Add unit cell if enabled
      if (showCell) {
        viewerInstance.addUnitCell();
      }
      
      // Center and zoom structure
      viewerInstance.center();
      viewerInstance.zoomTo();
      
      // Ensure proper camera positioning for centered view
      viewerInstance.render();
      
      setLoading(false);
      setError("");
    } catch (err) {
      setError(`Could not render: ${err.message}`);
      setLoading(false);
    }
  };

  // Get style configuration based on view mode - Optimized for performance
  const getStyleForViewMode = (mode, atomsVisible, bondsVisible) => {
    const baseStyle = {
      stick: bondsVisible ? {
        radius: 0.18, // ✅ Thinner for performance
        colorscheme: "Jmol", // ✅ Professional chemistry colors
      } : {},
      sphere: atomsVisible ? {
        scale: 0.34,
        colorscheme: "Jmol",
      } : {},
    };

    switch (mode) {
      case "stick":
        return {
          stick: bondsVisible ? { radius: 0.18, colorscheme: "Jmol" } : {},
          sphere: { scale: 0 }, // Hide spheres in stick mode
        };
      case "ball_stick":
        return baseStyle;
      case "space_filling":
        return {
          sphere: { scale: 0.95, colorscheme: "Jmol" },
        };
      default:
        return baseStyle;
    }
  };

  const handleResetCamera = () => {
    if (viewer.current) {
      viewer.current.zoomTo();
      viewer.current.render();
    }
  };

  const handleDownloadScreenshot = () => {
    try {
      if (viewer.current && viewer.current.canvas) {
        const canvas = viewer.current.canvas;
        const link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = `${formula.replace(/\s+/g, "_")}_structure.png`;
        link.click();
      }
    } catch (err) {
      setError(`Screenshot failed: ${err.message}`);
    }
  };

  const handleFullscreen = async () => {
    if (!viewerShell.current) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }
    await viewerShell.current.requestFullscreen();
  };

  // No structure message
  if (!activeCif) {
    return (
      <div style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "420px",
        background: "#f8fafc",
        border: "1px solid #e2e8f0",
        borderRadius: "12px",
        padding: "20px",
        textAlign: "center",
        color: "var(--muted)",
      }}>
        <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>⬡</div>
        <p style={{ margin: "0 0 4px 0", fontWeight: 600, fontSize: "1rem", color: "var(--text)" }}>
          No crystal structure available
        </p>
        <p style={{ margin: 0, fontSize: "0.85rem" }}>
          Generate a crystal to visualize it in 3D
        </p>
      </div>
    );
  }

  // Error state
  if (error && !loading) {
    return (
      <div style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "420px",
        background: "#fef2f2",
        border: "1px solid #fecaca",
        borderRadius: "12px",
        padding: "20px",
        textAlign: "center",
        color: "#991b1b",
      }}>
        <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>⚠</div>
        <p style={{ margin: "0 0 4px 0", fontWeight: 600, fontSize: "1rem" }}>
          Failed to render structure
        </p>
        <p style={{ margin: 0, fontSize: "0.85rem", color: "#7c2d12" }}>
          {error}
        </p>
        <p style={{ margin: "12px 0 0 0", fontSize: "0.8rem", fontStyle: "italic", color: "#7c2d12" }}>
          Check browser console for details
        </p>
      </div>
    );
  }

  // Render viewer with controls
  return (
  <div
    ref={viewerShell}
    className="embedded-viewer"
  >
    <div className="embedded-viewer__canvas-wrap">
      {loading && (
        <div className="embedded-viewer__loading">
          <div className="embedded-viewer__loader">◈</div>

          <div>
            <p>Rendering crystal structure...</p>
            <small>Preparing scientific visualization</small>
          </div>
        </div>
      )}

      <div
        ref={viewerContainer}
        className="embedded-viewer__canvas"
      />
    </div>

    {!loading && (
      <div className="embedded-viewer__toolbar">
        <button
          title="Reset camera"
          onClick={handleResetCamera}
          className={showCell ? "active" : ""}
        >
          🔄
        </button>

        <button
          title="Screenshot"
          onClick={handleDownloadScreenshot}
        >
          📸
        </button>

        <button
          title="Toggle atoms"
          onClick={() => setShowAtoms((value) => !value)}
          className={showAtoms ? "active" : ""}
        >
          ●
        </button>

        <button
          title="Toggle bonds"
          onClick={() => setShowBonds((value) => !value)}
          className={showBonds ? "active" : ""}
        >
          ━
        </button>

        <button
          title="Toggle unit cell"
          onClick={() => setShowCell((value) => !value)}
          className={showCell ? "active" : ""}
        >
          ▢
        </button>

        <button
          title="Auto rotate"
          onClick={() => setAutoRotate((value) => !value)}
          className={autoRotate ? "active" : ""}
        >
          ⟳
        </button>

        <button
          title="Fullscreen"
          onClick={handleFullscreen}
        >
          ⛶
        </button>
      </div>
    )}

    <style>{`
      .embedded-viewer {
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      .embedded-viewer__canvas-wrap {
        position: relative;
        width: 100%;

        aspect-ratio: 16 / 10;

        min-height: 460px;
        max-height: 560px;

        border-radius: 16px;
        overflow: hidden;

        background: #f8fafc;

        border: 1px solid #dbe4ee;
      }

      .embedded-viewer__canvas {
        width: 100%;
        height: 100%;
      }

      .embedded-viewer__toolbar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0px;

        // margin-top: 12px;
        padding: 10px;

        border: 1px solid #dbe4ee;
        border-radius: 12px;

        background: white;
      }

      .embedded-viewer__toolbar button {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        border: 1px solid rgba(148,163,184,0.16);
        background: #f8fafc;
        color: #475569;
        cursor: pointer;
        transition: 0.2s ease;
        font-size: 1rem;
      }

      .embedded-viewer__toolbar button:hover {
        background: #e2e8f0;
      }

      .embedded-viewer__toolbar button.active {
        background: #dbeafe;
        color: #2563eb;
        border-color: #93c5fd;
      }

      .embedded-viewer__loading {
        position: absolute;
        inset: 0;
        z-index: 20;

        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;

        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(10px);

        color: #0f172a;
      }

      .embedded-viewer__loader {
        font-size: 2.8rem;
        animation: spin 3s linear infinite;
      }

      @keyframes spin {
        from {
          transform: rotate(0deg);
        }

        to {
          transform: rotate(360deg);
        }
      }

      @media (max-width: 1280px) {
        .embedded-viewer__canvas-wrap {
          min-height: 420px;
          max-height: 520px;
        }
      }

      @media (max-width: 900px) {
        .embedded-viewer__canvas-wrap {
          min-height: 380px;
          max-height: 480px;
        }
      }

      @media (max-width: 680px) {
        .embedded-viewer__canvas-wrap {
          min-height: 340px;
          max-height: 420px;
        }

        .embedded-viewer__toolbar {
          flex-wrap: wrap;
          gap: 6px;
        }

        .embedded-viewer__toolbar button {
          width: 38px;
          height: 38px;
          font-size: 0.9rem;
        }
      }
    `}</style>
  </div>
);
}
