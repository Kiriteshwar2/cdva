import { useEffect, useRef, useState } from "react";

let viewerScriptPromise;

function load3DMolScript() {
  if (typeof window !== "undefined" && window.$3Dmol) {
    return Promise.resolve(window.$3Dmol);
  }

  if (!viewerScriptPromise) {
    viewerScriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js";
      script.async = true;
      script.onload = () => resolve(window.$3Dmol);
      script.onerror = () => reject(new Error("Failed to load 3Dmol.js"));
      document.body.appendChild(script);
    });
  }

  return viewerScriptPromise;
}

export default function CrystalViewer({ cifString, structureId }) {
  const containerRef = useRef(null);
  const [viewerError, setViewerError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function renderViewer() {
      if (!containerRef.current || !cifString) {
        return;
      }

      try {
        const $3Dmol = await load3DMolScript();
        if (cancelled || !containerRef.current) {
          return;
        }

        const viewer = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: "rgba(248,250,252,0.95)",
        });
        viewer.clear();
        viewer.addModel(cifString, "cif");
        viewer.setStyle({}, { stick: { radius: 0.16 }, sphere: { scale: 0.22 } });
        viewer.addUnitCell();
        viewer.zoomTo();
        viewer.render();
      } catch (error) {
        if (!cancelled) {
          setViewerError(error.message || "Viewer failed to initialize.");
        }
      }
    }

    renderViewer();

    return () => {
      cancelled = true;
    };
  }, [cifString, structureId]);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">3D viewer</span>
        {viewerError ? <span className="text-xs text-rose-600">{viewerError}</span> : null}
      </div>
      <div ref={containerRef} className="h-72 w-full" />
    </div>
  );
}
