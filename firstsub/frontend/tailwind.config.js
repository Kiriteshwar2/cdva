/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:      "#0a0e1a",
        card:    "#111827",
        border:  "#1f2d45",
        accent:  "#14b8a6",   // teal
        purple:  "#8b5cf6",
        glow:    "#06b6d4",
      },
      fontFamily: { sans: ["Inter", "sans-serif"] },
    },
  },
  plugins: [],
};
