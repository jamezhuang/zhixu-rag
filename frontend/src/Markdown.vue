<script setup>
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'
const props = defineProps({ text: { type: String, default: '' }, hideCitations: { type: Boolean, default: false } })
const emit = defineEmits(['source'])
const md = new MarkdownIt({ html: false, linkify: false, breaks: true })
md.renderer.rules.text = (tokens, idx) => {
  const text = md.utils.escapeHtml(tokens[idx].content)
  if (props.hideCitations) {
    return text.replace(/[ \t]*(?:\[\d+\][ \t]*)+(?=[，。；、！？：,.!?;:]|$)/g, '').replace(/\[\d+\]/g, '')
  }
  return text.replace(/\[(\d+)\]/g, '<button class="citation" data-cite="$1" aria-label="查看来源 $1">$1</button>')
}
const html = computed(() => DOMPurify.sanitize(md.render(props.text), { ADD_ATTR: ['data-cite', 'aria-label'] }))
function click(e) { const n = e.target.closest('[data-cite]'); if (n) emit('source', Number(n.dataset.cite)) }
</script>
<template><div class="markdown" v-html="html" @click="click" /></template>
