// LossChart.jsx — A line chart of training loss over epochs using recharts
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Legend,
} from "recharts";

export default function LossChart({ data }) {
  if (!data || data.length < 2) return null;
  // Sample every N points to avoid rendering thousands of dots
  const step   = Math.max(1, Math.floor(data.length / 100));
  const sample = data.filter((_, i) => i % step === 0);

  return (
    <div className="mt-2">
      <p className="label">Loss Curve</p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={sample}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2d45" />
          <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 10 }} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #1f2d45" }}
            labelStyle={{ color: "#9ca3af" }}
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#14b8a6"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
