// LogPanel.jsx — Scrolling log of all actions (upload, train, generate, validate)
import { useEffect, useRef } from "react";

export default function LogPanel({ logs }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="card space-y-2">
      <div className="flex justify-between items-center">
        <h2 className="text-accent font-bold text-lg">📋 Activity Log</h2>
        <span className="text-xs text-gray-500">{logs.length} events</span>
      </div>

      <div className="log-box">
        {logs.length === 0
          ? <span className="text-gray-600">No activity yet…</span>
          : logs.map((line, i) => (
            <div key={i} className="py-0.5">
              <span className="text-gray-600 mr-2 select-none">{String(i + 1).padStart(2, "0")}</span>
              {line}
            </div>
          ))
        }
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
