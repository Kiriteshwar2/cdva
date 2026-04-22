// App.jsx — Main application layout wiring all panels together
import { useState } from "react";
import UploadPanel     from "./components/UploadPanel";
import TrainPanel      from "./components/TrainPanel";
import GeneratePanel   from "./components/GeneratePanel";
import OutputPanel     from "./components/OutputPanel";
import ValidationPanel from "./components/ValidationPanel";
import MultiGenPanel   from "./components/MultiGenPanel";
import LogPanel        from "./components/LogPanel";

export default function App() {
  const [generated, setGenerated] = useState(null);
  const [antiScore, setAntiScore] = useState(0);
  const [logs, setLogs]           = useState([]);

  const addLog = (msg) =>
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handleGenerate = (data) => {
    setGenerated(data);
    setAntiScore(data.antigravity_score);
  };

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "var(--bg)" }}>

      {/* ── Header ── */}
      <header style={{
        borderBottom: "1px solid var(--border)",
        backgroundColor: "rgba(17,24,39,0.9)",
        backdropFilter: "blur(8px)",
        position: "sticky", top: 0, zIndex: 50,
      }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0.75rem 1rem",
                      display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div style={{
              width: 40, height: 40, borderRadius: 10,
              background: "linear-gradient(135deg,#14b8a6,#8b5cf6)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 20, fontWeight: 900, color: "#000",
            }}>⬡</div>
            <div>
              <div style={{ color: "#fff", fontWeight: 700, fontSize: "1.1rem", lineHeight: 1.2 }}>
                CDVAE Crystal Generator
              </div>
              <div style={{ color: "#9ca3af", fontSize: "0.7rem" }}>
                Crystal Diffusion Variational Autoencoder
              </div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.75rem", color: "#9ca3af" }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              backgroundColor: "#14b8a6", display: "inline-block",
              animation: "pulse 2s infinite",
            }}/>
            Backend · localhost:8000
          </div>
        </div>
      </header>

      {/* ── Content ── */}
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "1.5rem 1rem" }}>

        {/* Row 1: Upload + Log */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem",
                      marginBottom: "1.5rem" }}
             className="grid-2col">
          <UploadPanel onLog={addLog} />
          <LogPanel    logs={logs} />
        </div>

        {/* Row 2: Train */}
        <div style={{ marginBottom: "1.5rem" }}>
          <TrainPanel onLog={addLog} />
        </div>

        {/* Row 3: Generate + Output */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem",
                      marginBottom: "1.5rem" }}
             className="grid-2col">
          <GeneratePanel onGenerate={handleGenerate} onLog={addLog} />
          <OutputPanel   data={generated} />
        </div>

        {/* Row 4: Validation */}
        <div style={{ marginBottom: "1.5rem" }}>
          <ValidationPanel generatedData={generated} onLog={addLog} />
        </div>

        {/* Row 5: Multi-gen */}
        <div style={{ marginBottom: "1.5rem" }}>
          <MultiGenPanel antigravityScore={antiScore} onLog={addLog} />
        </div>
      </main>

      {/* ── Footer ── */}
      <footer style={{ borderTop: "1px solid var(--border)", padding: "1rem",
                       textAlign: "center", fontSize: "0.75rem", color: "#4b5563" }}>
        CDVAE Crystal Generator · FastAPI + React · CPU-friendly
      </footer>

      {/* Responsive grid override */}
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @media (max-width: 768px) {
          .grid-2col { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
