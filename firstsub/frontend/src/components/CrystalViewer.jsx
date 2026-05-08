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

export default function CrystalViewer({ cifString, structureId, bonds = [] }) {
  const containerRef = useRef(null);
  const [viewerError, setViewerError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function renderViewer() {
      if (!containerRef.current || !cifString) {
        return;
      }

      try {
        setViewerError("");
        const $3Dmol = await load3DMolScript();
        if (cancelled || !containerRef.current) {
          return;
        }

        containerRef.current.style.minHeight = "360px";
        containerRef.current.style.width = "100%";
        containerRef.current.innerHTML = "";

        const viewer = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: "rgba(248,250,252,0.95)",
        });
        viewer.clear();
        const model = viewer.addModel(cifString, "cif");

        if (model && typeof model.selectedAtoms === "function") {
          const atoms = model.selectedAtoms();
          const MAX_BOND_DISTANCE = 3.8;

          for (let i = 0; i < atoms.length; i += 1) {
            for (let j = i + 1; j < atoms.length; j += 1) {
              const dx = atoms[i].x - atoms[j].x;
              const dy = atoms[i].y - atoms[j].y;
              const dz = atoms[i].z - atoms[j].z;

              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

              if (dist < MAX_BOND_DISTANCE && typeof model.addBond === "function") {
                model.addBond(i, j);
              }
            }
          }
        }

        if (Array.isArray(bonds) && bonds.length > 0) {
          for (const pair of bonds) {
            if (!Array.isArray(pair) || pair.length !== 2) {
              continue;
            }
            const [atom1, atom2] = pair;
            if (!Number.isInteger(atom1) || !Number.isInteger(atom2) || atom1 === atom2) {
              continue;
            }

            if (typeof viewer.addBond === "function") {
              viewer.addBond(atom1, atom2);
            } else if (model && typeof model.addBond === "function") {
              model.addBond(atom1, atom2);
            }
          }
        } else {
          // Fallback for legacy payloads that do not include chemistry-aware bonds.
          if (model && typeof model.setBonding === "function") {
            model.setBonding(true);
          } else if (typeof viewer.getModel === "function") {
            const firstModel = viewer.getModel(0);
            if (firstModel && typeof firstModel.setBonding === "function") {
              firstModel.setBonding(true);
            }
          }
        }

        viewer.setStyle({}, {
          stick: {
            radius: 0.12,
          },
          sphere: {
            scale: 0.20,
          },
        });

        // Extra stick pass helps when default style application misses inferred bonds.
        viewer.addStyle({}, { stick: {} });
        viewer.addUnitCell();
        viewer.zoomTo();
        viewer.render();
        requestAnimationFrame(() => viewer.resize());
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
  }, [cifString, structureId, bonds]);

  return (
    <div className="crystal-viewer">
      <div ref={containerRef} className="crystal-viewer__canvas" style={{ minHeight: 360 }} />
      {viewerError ? <div className="crystal-viewer__error">{viewerError}</div> : null}
    </div>
  );
}
