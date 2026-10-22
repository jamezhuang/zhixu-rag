import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './tests', timeout: 60000, fullyParallel: false,
  use: { baseURL: 'http://127.0.0.1:5173', headless: true, screenshot: 'only-on-failure' },
  reporter: [['list']],
})
