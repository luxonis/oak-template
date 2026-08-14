import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '',
  plugins: [react()],
  define: {
    global: {},
  },
  worker: {
    format: 'es',
  },
  build: {
    rollupOptions: {
      output: {
        format: 'esm',
      },
    },
  },
});
