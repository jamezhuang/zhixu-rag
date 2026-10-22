export async function api(path, options = {}) {
  const token = sessionStorage.getItem('zhixu_token')
  const headers = { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...options.headers }
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
    options.body = JSON.stringify(options.body)
  }
  let response
  try { response = await fetch(`/api${path}`, { ...options, headers }) }
  catch { throw new Error('无法连接服务，请确认后端已启动') }
  let data
  try { data = await response.json() } catch { throw new Error('服务暂不可用，请稍后重试') }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : '请检查填写内容后重试'
    if (response.status === 401 && !path.includes('/login')) window.dispatchEvent(new Event('session-expired'))
    throw new Error(detail)
  }
  return data
}
