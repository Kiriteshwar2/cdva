// LogPanel.jsx — Scrolling log of all actions
import { useEffect, useRef } from "react";

export default function LogPanel({ logs }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span className="card-title-icon">☰</span>
          Training Logs
        </div>
        <span style={{
          fontSize: "0.72rem", fontWeight: 600,
          color: "var(--text-light)",
          background: "var(--content-bg)",
          border: "1px solid var(--border)",
          padding: "2px 8px", borderRadius: 20,
        }}>
          {logs.length} events
        </span>
      </div>

      <div className="log-box" style={{ flex: 1 }}>
        {logs.length === 0 ? (
          <span style={{ color: "#334155" }}>No activity yet…</span>
        ) : (
          logs.map((line, i) => (
            <div key={i}>
              <span className="log-line-num">{String(i + 1).padStart(2, "0")}</span>
              {line}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
