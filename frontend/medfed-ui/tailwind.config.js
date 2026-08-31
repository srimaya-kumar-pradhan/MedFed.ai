/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Strict grayscale + single accent (clinical teal).
        ink: {
          900: "#0A0A0A",
          800: "#1A1A1A",
          700: "#2C2C2C",
          500: "#4A4A4A",
          400: "#8A8A8A",
          300: "#B8B8B8",
          200: "#E5E5E5",
          100: "#F2F2F2",
          50:  "#FAFAFA",
        },
        paper: "#FFFFFF",
        accent: {
          DEFAULT: "#0F4C5C", // deep clinical teal
          50: "#E6EFF1",
          100: "#C2D6DC",
          600: "#0C3D4A",
          700: "#082C36",
        },
        status: {
          good: "#0F4C5C",
          warn: "#8A6A1F",
          bad:  "#8A2A2A",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        // Consistent type scale.
        xs: ["11px", "16px"],
        sm: ["13px", "20px"],
        base: ["15px", "24px"],
        lg: ["17px", "26px"],
        xl: ["20px", "28px"],
        "2xl": ["24px", "32px"],
        "3xl": ["30px", "38px"],
        "4xl": ["36px", "44px"],
      },
      spacing: {
        // 8px grid.
        "0.5": "4px",
        "1":   "8px",
        "2":   "16px",
        "3":   "24px",
        "4":   "32px",
        "5":   "40px",
        "6":   "48px",
        "8":   "64px",
        "10":  "80px",
        "12":  "96px",
      },
      borderRadius: {
        none: "0",
        sm:   "2px",
        DEFAULT: "3px",
        md:   "4px",
        lg:   "6px",
      },
      transitionDuration: {
        DEFAULT: "150ms",
        slow: "200ms",
      },
    },
  },
  plugins: [],
};
