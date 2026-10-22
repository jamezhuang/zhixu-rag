import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
export default defineConfig({ plugins: [vue()], server: { port: 5173, strictPort: true, proxy: { '/api': { target: process.env.API_PROXY || 'http://127.0.0.1:8000', changeOrigin: true } } }, preview: { port: 5173 } })
