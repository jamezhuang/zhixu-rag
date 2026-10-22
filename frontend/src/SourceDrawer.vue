<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { X, FileText, ArrowUpRight, CalendarDays } from 'lucide-vue-next'
import { api } from './api'
import Markdown from './Markdown.vue'
const props = defineProps({ source: Object })
const emit = defineEmits(['close'])
const full = ref(null), error = ref(''), loading = ref(false), panel = ref(null)
let returnFocus
watch(() => props.source, () => { full.value = null; error.value = ''; setTimeout(() => panel.value?.focus(), 0) })
async function load() { loading.value = true; try { full.value = await api(`/documents/${props.source.document_id}`) } catch (e) { error.value = e.message } finally { loading.value = false } }
function key(e) {
  if (e.key === 'Escape') emit('close')
  if (e.key === 'Tab' && panel.value) {
    const nodes = [...panel.value.querySelectorAll('button,a,input')].filter(n => !n.disabled)
    const first = nodes[0], last = nodes.at(-1)
    if (e.shiftKey && (document.activeElement === first || document.activeElement === panel.value)) { e.preventDefault(); last?.focus() }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus() }
  }
}
onMounted(() => { returnFocus = document.activeElement; window.addEventListener('keydown', key); panel.value?.focus() })
onUnmounted(() => { window.removeEventListener('keydown', key); returnFocus?.focus() })
</script>
<template>
  <div class="drawer-backdrop" @click.self="emit('close')">
    <aside ref="panel" class="source-drawer" role="dialog" aria-modal="true" aria-labelledby="source-title" tabindex="-1">
      <header><span class="eyebrow">REFERENCE / 来源核对</span><button class="icon-button" aria-label="关闭来源" @click="emit('close')"><X :size="20"/></button></header>
      <div class="file-symbol"><FileText :size="26"/></div>
      <h2 id="source-title">{{ source.title }}</h2>
      <p class="muted">{{ source.code }} <span class="dot">·</span> v{{ source.version }}</p>
      <div class="source-meta"><CalendarDays :size="16"/> 生效 {{ source.effective_from }} <span v-if="source.effective_to">至 {{ source.effective_to }}（不含）</span></div>
      <div class="section-label">引用条款 · {{ source.section }}</div>
      <blockquote>{{ source.text }}</blockquote>
      <small class="muted">原文位置：{{ source.location || '正文' }} · 所有制度均为模拟资料</small>
      <button class="button outline full" :disabled="loading" @click="load">{{ loading ? '读取中…' : '查看这个版本的全文' }}<ArrowUpRight :size="16"/></button>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <section v-if="full" class="full-document"><span v-if="full.status === 'disabled'" class="notice">此版本已停用，此处展示历史引用原文。</span><Markdown :text="full.content"/></section>
    </aside>
  </div>
</template>
