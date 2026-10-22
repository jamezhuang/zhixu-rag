// Suggestions use all saved versions, including drafts and disabled records.
export function suggestCode(docs, prefix) {
  const numbers = docs.map(d => d.code).filter(code => code.startsWith(`${prefix}-`))
    .map(code => Number(code.split('-')[1])).filter(Number.isFinite)
  const next = Math.max(0, ...numbers) + 1
  return next <= 999 ? `${prefix}-${String(next).padStart(3, '0')}` : ''
}

export function versionsFor(docs, code) {
  return docs.filter(d => d.code === code.trim().toUpperCase()).sort((a, b) =>
    b.version.localeCompare(a.version, undefined, { numeric: true }) || b.effective_from.localeCompare(a.effective_from))
}

export function suggestVersion(docs, code) {
  const versions = versionsFor(docs, code)
  if (!versions.length) return '1.0'
  if (versions.some(d => !/^\d+\.\d+$/.test(d.version))) return ''
  return `${Math.max(...versions.map(d => Number(d.version.split('.')[0]))) + 1}.0`
}
