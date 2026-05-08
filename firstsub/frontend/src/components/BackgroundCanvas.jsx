// BackgroundCanvas.jsx — Animated crystal lattice particle background
import { useEffect, useRef } from "react";

const PARTICLE_COUNT = 55;
const CONNECTION_DIST = 130;
const COLORS = ["#5b5ef4", "#818cf8", "#6366f1", "#a5b4fc", "#4f46e5"];

function randomBetween(a, b) {
  return a + Math.random() * (b - a);
}

export default function BackgroundCanvas() {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const particles = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Initialise particles
    particles.current = Array.from({ length: PARTICLE_COUNT }, () => ({
      x:    randomBetween(0, canvas.width),
      y:    randomBetween(0, canvas.height),
      vx:   randomBetween(-0.25, 0.25),
      vy:   randomBetween(-0.25, 0.25),
      r:    randomBetween(1.5, 3.5),
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      pulse: randomBetween(0, Math.PI * 2),
      pulseSpeed: randomBetween(0.008, 0.022),
    }));

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;

      ctx.clearRect(0, 0, W, H);

      // Subtle radial gradient overlay (deepens the background feel)
      const grad = ctx.createRadialGradient(W * 0.5, H * 0.4, 0, W * 0.5, H * 0.4, Math.max(W, H) * 0.75);
      grad.addColorStop(0, "rgba(91,94,244,0.04)");
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);

      const pts = particles.current;

      // Update & draw connections
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        // Move
        p.x += p.vx;
        p.y += p.vy;
        p.pulse += p.pulseSpeed;

        // Wrap around edges
        if (p.x < 0)  p.x = W;
        if (p.x > W)  p.x = 0;
        if (p.y < 0)  p.y = H;
        if (p.y > H)  p.y = 0;

        // Draw connections to neighbours
        for (let j = i + 1; j < pts.length; j++) {
          const q    = pts[j];
          const dist = Math.hypot(p.x - q.x, p.y - q.y);
          if (dist < CONNECTION_DIST) {
            const alpha = (1 - dist / CONNECTION_DIST) * 0.18;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = `rgba(91,94,244,${alpha})`;
            ctx.lineWidth   = 0.8;
            ctx.stroke();
          }
        }
      }

      // Draw particles on top
      for (const p of pts) {
        const glow = 0.7 + 0.3 * Math.sin(p.pulse);
        // Outer glow
        const rg = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 4);
        rg.addColorStop(0, `rgba(91,94,244,${0.12 * glow})`);
        rg.addColorStop(1, "rgba(91,94,244,0)");
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 4, 0, Math.PI * 2);
        ctx.fillStyle = rg;
        ctx.fill();

        // Core dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * glow, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 0.55 * glow;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      animRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 0,
      }}
    />
  );
}
