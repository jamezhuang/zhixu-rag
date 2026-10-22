import { test, expect } from '@playwright/test'

async function login(page, role = '运营员工') {
  await page.goto('/')
  await page.getByRole('button', { name: role, exact: true }).click()
  await page.getByRole('button', { name: '进入工作台' }).click()
}

test('employee asks, verifies original source, and leaves feedback', async ({ page }) => {
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await login(page)
  await expect(page.getByRole('heading', { name: /知识助手\s*有据可循/ })).toBeVisible()
  await page.getByLabel('向知识助手提问').fill('试用期运营能直接确认促销折扣吗？不能的话找谁？')
  await page.getByRole('button', { name: '发送问题', exact: true }).click()
  await expect(page.locator('.assistant-message')).toBeVisible({ timeout: 30000 })
  await expect(page.locator('.assistant-message')).toContainText('原文摘录')
  await page.locator('.answer-sources button').first().click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.getByRole('button', { name: '查看这个版本的全文' }).click()
  await expect(page.locator('.full-document')).toContainText('模拟')
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await page.getByRole('button', { name: '回答有帮助' }).click()
  await expect(page.getByText('谢谢，你的反馈已记录')).toBeVisible()
  await expect(page.locator('body')).not.toContainText('undefined')
  expect(errors).toEqual([])
})

test('administrator can inspect documents and run details', async ({ page }) => {
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await login(page, '管理员')
  await expect(page.getByRole('heading', { name: '工作台概览' })).toBeVisible()
  await page.getByRole('button', { name: '知识文档', exact: true }).click()
  await expect(page.getByRole('table')).toBeVisible()
  await page.getByLabel('搜索文档').fill('试用期')
  await expect(page.locator('tbody tr')).toHaveCount(2)
  await page.getByRole('button', { name: '预览试用期工作与授权规定' }).first().click()
  await expect(page.getByRole('dialog')).toContainText('试用期')
  await page.getByRole('button', { name: '关闭窗口' }).click()
  await page.getByRole('button', { name: '员工管理', exact: true }).click()
  await expect(page.locator('.employee-card')).toHaveCount(3)
  await page.getByRole('button', { name: '问答与反馈', exact: true }).click()
  await expect(page.getByRole('table')).toBeVisible()
  expect(errors).toEqual([])
})

test('mobile employee workspace fits viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await login(page)
  await expect(page.getByRole('button', { name: '展开导航' })).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
  expect(overflow).toBe(false)
  await page.getByRole('button', { name: '展开导航' }).click()
  await page.getByRole('button', { name: '知识文库', exact: true }).click()
  await expect(page.getByRole('heading', { name: '知识文库' })).toBeVisible()
})
