// LossChart.jsx — A line chart of training loss over epochs using recharts
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer,
} from "recharts";

export default function LossChart({ data }) {
  if (!data || data.length < 2) return null;
  const step   = Math.max(1, Math.floor(data.length / 100));
  const sample = data.filter((_, i) => i % step === 0);

  return (
    <div>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={sample} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="epoch"
            stroke="#94a3b8"
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            label={{ value: "Epoch", position: "insideBottom", offset: -2, fontSize: 10, fill: "#94a3b8" }}
          />
          <YAxis
            stroke="#94a3b8"
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            label={{ value: "Loss", angle: -90, position: "insideLeft", offset: 16, fontSize: 10, fill: "#94a3b8" }}
            domain={["auto", "auto"]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 8,
              fontSize: "0.75rem",
              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
            }}
            labelStyle={{ color: "#64748b" }}
            itemStyle={{ color: "#5b5ef4" }}
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#5b5ef4"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
