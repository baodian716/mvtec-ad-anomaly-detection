import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 開發時把 /api 轉給 FastAPI 後端（demo/backend/server.py，port 7860）
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:7860' } },
})
