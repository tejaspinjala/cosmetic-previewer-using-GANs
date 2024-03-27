/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primaryText: "#563517",
        primaryLightPink: "#DED7D3",
      },
    },
  },
  plugins: [],
};
