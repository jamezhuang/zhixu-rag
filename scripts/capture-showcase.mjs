import { chromium } from '../frontend/node_modules/@playwright/test/index.mjs'
import { mkdir, readFile } from 'node:fs/promises'

// Read the local password without logging it or writing tokens into artifacts.
const env = await readFile('.env', 'utf8')
const password = process.env.ADMIN_PASSWORD || env.match(/^ADMIN_PASSWORD\s*=\s*(.*)$/m)?.[1]?.trim().replace(/^(['"])(.*)\1$/, '$2')
if (!password) throw new Error('Configure ADMIN_PASSWORD locally before capture')
const response = await fetch('http://127.0.0.1:8000/api/auth/login', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'admin', password }),
})
if (!response.ok) throw new Error('Administrator login failed')
const { token } = await response.json()
const headers = { Authorization: `Bearer ${token}` }
const conversations = await (await fetch('http://127.0.0.1:8000/api/conversations', { headers })).json()
let chosen
for (const c of conversations) {
  const detail = await (await fetch(`http://127.0.0.1:8000/api/conversations/${c.id}`, { headers })).json()
  if (detail.messages?.some(m => m.role === 'assistant' && m.sources?.length) &&
      detail.messages?.some(m => m.role === 'user' && m.content.includes('导师复核后'))) {
    chosen = c.id
    break
  }
}
await mkdir('docs/assets', { recursive: true })
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 })
  await page.addInitScript(value => sessionStorage.setItem('zhixu_token', value), token)
  await page.goto('http://127.0.0.1:5173/#chat')
  await page.getByRole('heading', { name: /知识助手/ }).waitFor()
  await page.screenshot({ path: 'docs/assets/workspace.png', fullPage: false })
  if (chosen) {
    await page.goto(`http://127.0.0.1:5173/?conversation=${chosen}#chat`)
    await page.locator('.assistant-message').first().waitFor()
    await page.locator('.thread').evaluate(el => { el.style.scrollBehavior = 'auto'; el.scrollTop = 0 })
    await page.waitForTimeout(500)
    await page.screenshot({ path: 'docs/assets/answer.png', fullPage: false })
    await page.locator('.thread').evaluate(el => { el.scrollTop = el.scrollHeight })
    await page.waitForTimeout(300)
    await page.screenshot({ path: 'docs/assets/answer-sources.png', fullPage: false })
  }
  await page.getByRole('button', { name: '知识文档', exact: true }).click()
  await page.locator('tbody tr').first().waitFor()
  await page.screenshot({ path: 'docs/assets/documents.png', fullPage: false })
  console.log(JSON.stringify({ screenshots: ['workspace.png', ...(chosen ? ['answer.png', 'answer-sources.png'] : []), 'documents.png'], source: 'local live application' }))
} finally {
  await browser.close()
}
