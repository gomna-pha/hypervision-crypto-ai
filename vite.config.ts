import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: 'index.html'
      }
    }
  },
  base: '/hypervision-crypto-ai/',
  server: {
    port: 3000
  }
})