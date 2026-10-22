<script setup>
import { ref, computed, nextTick, watch, onMounted } from 'vue'
import { ArrowUp, ArrowUpRight, BookOpen, FileText, Sparkles, ThumbsUp, ThumbsDown, RotateCcw, Plus, MessageSquare, ChevronRight, LoaderCircle, X, Square } from 'lucide-vue-next'
import { api } from './api'
import Markdown from './Markdown.vue'
import SourceDrawer from './SourceDrawer.vue'
const props = defineProps({ user: Object, mode: String, initialConversation: String })
const conversations = ref([]), selected = ref(''), messages = ref([]), runs = ref([]), question = ref(''), busy = ref(false), error = ref(''), source = ref(null), thread = ref(null), historyOpen = ref(false), rated = ref({}), feedbackRun = ref(null), feedbackReason = ref('')
const suggestions = [
  { title: '从这里认识公司', text: '公司为品牌客户提供哪些服务？', label: '公司业务', icon: BookOpen },
  { title: '把第一周安排明白', text: '我是新入职的运营，第一周应该学习什么？', label: '入职指南', icon: Sparkles },
  { title: '弄清楚权限与协作', text: '试用期运营能直接确认促销折扣吗？不能的话找谁？', label: '流程与审批', icon: FileText },
]
const pending = computed(() => [...runs.value].reverse().find(r => ['pending', 'running', 'failed'].includes(r.status)))
const activeRun = ref(''), stopping = ref(false)
const lastStopped = computed(() => runs.value.at(-1)?.status === 'cancelled')
async function stopGeneration() {
  const id = activeRun.value || pending.value?.id
  if (!id || stopping.value) return
  stopping.value = true
  try {
    await api(`/runs/${id}/cancel`, { method: 'POST' })
  } catch (e) { error.value = e.message; stopping.value = false }
  if (!busy.value) { stopping.value = false; await select(selected.value) }
}
async function loadList() { conversations.value = await api('/conversations') }
async function select(id) {
  if (busy.value) return
  selected.value = id; error.value = ''; historyOpen.value = false
  try { const data = await api(`/conversations/${id}`); messages.value = data.messages; runs.value = data.runs; await scroll() } catch (e) { error.value = e.message }
}
function fresh() { if (busy.value) return; selected.value = ''; messages.value = []; runs.value = []; error.value = ''; question.value = ''; historyOpen.value = false }
async function scroll() { await nextTick(); thread.value?.scrollTo({ top: thread.value.scrollHeight, behavior: 'smooth' }) }
async function execute(run) {
  busy.value = true; error.value = ''
  activeRun.value = run.id
  try {
    await api(`/runs/${run.id}/execute`, { method: 'POST' })
  } catch (e) { error.value = e.message }
  finally { busy.value = false; activeRun.value = ''; stopping.value = false; if (selected.value) await select(selected.value); await loadList() }
}
async function send(text = question.value) {
  text = text.trim()
  if (!text || busy.value || pending.value) return
  busy.value = true; error.value = ''
  try {
    if (!selected.value) { const c = await api('/conversations', { method: 'POST' }); selected.value = c.id }
    const run = await api(`/conversations/${selected.value}/requests`, { method: 'POST', body: { question: text, request_id: crypto.randomUUID() } })
    activeRun.value = run.id
    question.value = ''; messages.value.push({ role: 'user', content: text, request_id: run.id }); await scroll()
    await api(`/runs/${run.id}/execute`, { method: 'POST' })
  } catch (e) { error.value = e.message }
  finally { busy.value = false; activeRun.value = ''; stopping.value = false; if (selected.value) await select(selected.value); await loadList() }
}
async function abandon() { try { await api(`/runs/${pending.value.id}/cancel`, { method: 'POST' }); await select(selected.value) } catch (e) { error.value = e.message } }
async function feedback(id, helpful) {
  if (!helpful && feedbackRun.value !== id) { feedbackRun.value = id; feedbackReason.value = ''; return }
  try { await api(`/runs/${id}/feedback`, { method: 'POST', body: { helpful, reason: helpful ? '' : feedbackReason.value } }); rated.value[id] = helpful; feedbackRun.value = null } catch (e) { error.value = e.message }
}
onMounted(async () => { try { await loadList(); if (props.initialConversation) await select(props.initialConversation) } catch (e) { error.value = e.message } })
</script>
<template>
  <div class="chat-layout">
    <section class="chat-column">
      <header class="page-header"><div><span class="eyebrow">YOUR KNOWLEDGE COMPANION</span><h1>知识助手<span class="pill quiet">有据可循</span></h1></div><div class="header-actions"><button class="button outline compact" @click="historyOpen = !historyOpen"><MessageSquare :size="16"/>历史对话</button><button class="icon-button" aria-label="新建对话" :disabled="busy" @click="fresh"><Plus :size="20"/></button></div></header>
      <div v-if="historyOpen" class="history-panel"><div class="row-between"><strong>我的对话</strong><button class="icon-button" aria-label="关闭历史对话" @click="historyOpen = false"><X :size="16"/></button></div><div class="history-list" tabindex="0" role="region" aria-label="历史对话列表"><p v-if="!conversations.length" class="muted">还没有历史对话</p><button v-for="c in conversations" :key="c.id" class="history-item" :class="{ selected: c.id === selected }" @click="select(c.id)"><MessageSquare :size="15"/><span>{{ c.title }}</span><ChevronRight :size="14"/></button></div></div>
      <div ref="thread" class="thread" :class="{ empty: !messages.length }" aria-live="polite" aria-busy="busy">
        <div v-if="!messages.length" class="welcome">
          <div class="welcome-symbol"><BookOpen :size="28" :stroke-width="1.5"/></div>
          <span class="eyebrow">少一点摸索，多一点笃定</span>
          <h2>{{ user.name.split(' ')[0] }}，很高兴与你一起开始。</h2>
          <p>关于公司、业务和你的新角色，<br/>把问题交给我，把时间留给成长。</p>
          <div class="suggestion-grid"><button v-for="s in suggestions" :key="s.title" class="suggestion" @click="send(s.text)"><span class="suggestion-icon"><component :is="s.icon" :size="20" :stroke-width="1.6"/></span><span class="suggestion-label">{{ s.label }}</span><strong>{{ s.title }}</strong><span class="suggestion-text">{{ s.text }}</span><ArrowUpRight :size="17" class="suggestion-arrow"/></button></div>
          <div class="welcome-note"><span class="status-dot"/>从已发布的公司文档中寻找答案，并为你标明出处</div>
        </div>
        <template v-for="(m, i) in messages" :key="m.id || i">
          <div v-if="m.role === 'user'" class="user-message"><span>{{ m.content }}</span><div class="avatar small">{{ user.name[0] }}</div></div>
          <article v-else class="assistant-message"><div class="assistant-avatar"><BookOpen :size="18"/></div><div class="answer-body"><div class="answer-byline">知序 <span>知识助手</span></div><Markdown :text="m.content" hide-citations @source="n => source = m.sources.find(s => s.citation === n) || null"/><div v-if="m.sources?.length" class="answer-sources"><span>参考来源</span><button v-for="s in m.sources.slice(0, 6)" :key="s.id" @click="source = s"><FileText :size="13"/>{{ s.title }}<small>v{{ s.version }}</small></button><button v-if="m.sources.length > 6" @click="source = m.sources[6]">更多 {{ m.sources.length - 6 }} 项</button></div><div class="feedback-row"><span v-if="m.request_id in rated">谢谢，你的反馈已记录</span><template v-else><span>这个回答有帮助吗？</span><button class="icon-button" aria-label="回答有帮助" @click="feedback(m.request_id, true)"><ThumbsUp :size="15"/></button><button class="icon-button" aria-label="回答无帮助" @click="feedback(m.request_id, false)"><ThumbsDown :size="15"/></button></template></div><form v-if="feedbackRun === m.request_id" class="feedback-form" @submit.prevent="feedback(m.request_id, false)"><label :for="`reason-${i}`">可以告诉我们哪里需要改进（选填）</label><input :id="`reason-${i}`" v-model="feedbackReason" maxlength="1000" placeholder="例如：遗漏了试用期的特殊要求"/><button class="button outline compact">提交反馈</button></form></div></article>
        </template>
        <div v-if="busy" class="thinking" role="status"><LoaderCircle :size="18" class="spin"/>{{ stopping ? '正在停止生成…' : '正在查阅资料并核对依据…' }}</div>
        <p v-else-if="lastStopped" class="muted" role="status">已停止生成，可以继续提问。</p>
        <div v-if="pending && !busy" class="run-error"><strong>{{ pending.status === 'failed' ? '这次回答暂未完成' : '上一次问题还未完成' }}</strong><p>{{ pending.error || '可以继续执行或放弃该问题。' }}</p><div><button class="button outline compact" @click="execute(pending)"><RotateCcw :size="15"/>重试本次问题</button><button class="text-button" @click="abandon">放弃问题</button></div></div>
      </div>
      <div class="composer-area"><p v-if="error" class="error" role="alert">{{ error }}</p><form class="composer" @submit.prevent="send()"><label class="sr-only" for="question">向知识助手提问</label><textarea id="question" v-model="question" placeholder="问问公司业务、岗位职责或办事流程…" rows="2" maxlength="4000" :disabled="busy || !!pending" @keydown.enter.exact="e => { if (!e.isComposing) { e.preventDefault(); send() } }"/><div class="composer-bottom"><span><BookOpen :size="14"/>公司知识库 <span class="composer-separator">/</span> {{ mode === 'demo' ? '关键词演示' : '检索增强' }}</span><button class="send-button" type="button" :aria-label="busy ? (stopping ? '正在停止生成' : '停止生成') : '发送问题'" :title="busy ? '停止生成' : '发送问题'" :disabled="busy ? (!activeRun || stopping) : (!question.trim() || !!pending)" @click="busy ? stopGeneration() : send()"><LoaderCircle v-if="busy && (!activeRun || stopping)" :size="19" class="spin"/><Square v-else-if="busy" :size="17" fill="currentColor"/><ArrowUp v-else :size="20"/></button></div></form><div class="composer-hint"><span>Enter 发送 · Shift + Enter 换行</span><span>重要信息请核对引用原文</span></div></div>
    </section>
    <aside class="context-column"><div class="context-heading">你的入职工作台<span class="pill">NEW</span></div><div class="profile-card"><div class="profile-top"><div class="avatar">{{ user.name[0] }}</div><div><strong>{{ user.name }}</strong><p>{{ user.department }} · {{ user.position }}</p></div></div><div class="profile-detail"><span>员工状态</span><strong>{{ user.probation ? '试用期' : '正式员工' }}</strong></div><div class="profile-detail"><span>入职日期</span><strong>{{ user.hire_date }}</strong></div></div><div class="context-section"><span class="eyebrow">GETTING STARTED</span><h3>从了解，到上手。</h3><p>围绕你的新角色，逐步建立对业务的全貌。</p><button v-for="(label, i) in ['了解我们提供的服务', '认识职责与协作关系', '熟悉项目与活动流程']" :key="label" class="learning-step" :disabled="busy || !!pending" @click="send(['公司为品牌客户提供哪些服务？', '运营、客服、销售和投放组分别负责什么？', '客户项目从签约到启动有哪些步骤？'][i])"><span>{{ String(i + 1).padStart(2, '0') }}</span><strong>{{ label }}</strong><ChevronRight :size="16"/></button></div><div class="tip-card"><span class="tip-mark">“</span><p>不用一次问完整。<br/>你可以接着问「那我呢？」<br/>也可以告诉我具体的条件。</p><span>让知识，接得住每一次追问。</span></div><div class="context-footer"><span class="status-dot"/>模拟企业知识 · 学习与演示用途</div></aside>
    <SourceDrawer v-if="source" :source="source" @close="source = null"/>
  </div>
</template>
