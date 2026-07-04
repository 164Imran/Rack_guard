/** @type {import('tailwindcss').Config} */
const config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif"
        ]
      },
      boxShadow: {
        calm: "0 24px 80px rgba(0, 0, 0, 0.34)",
        glow: "0 0 50px rgba(67, 199, 232, 0.16)"
      }
    }
  },
  plugins: []
};

export default config;
