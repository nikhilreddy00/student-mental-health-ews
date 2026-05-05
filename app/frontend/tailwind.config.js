/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Sora', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
    },
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: [
      {
        signal: {
          'color-scheme': 'dark',
          primary: '#38bdf8',
          'primary-content': '#0c0f1a',
          secondary: '#818cf8',
          'secondary-content': '#ffffff',
          accent: '#34d399',
          'accent-content': '#0c0f1a',
          neutral: '#1e293b',
          'neutral-content': '#94a3b8',
          'base-100': '#0c0f1a',
          'base-200': '#111827',
          'base-300': '#1a2235',
          'base-content': '#e2e8f0',
          info: '#38bdf8',
          'info-content': '#0c0f1a',
          success: '#34d399',
          'success-content': '#0c0f1a',
          warning: '#fb923c',
          'warning-content': '#0c0f1a',
          error: '#f43f5e',
          'error-content': '#ffffff',
        },
      },
    ],
    darkTheme: 'signal',
    base: true,
    logs: false,
  },
}
