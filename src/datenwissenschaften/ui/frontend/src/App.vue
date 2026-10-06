<script setup>
import hljs from 'highlight.js/lib/core'
import dockerfile from 'highlight.js/lib/languages/dockerfile'
import ini from 'highlight.js/lib/languages/ini'
import plaintext from 'highlight.js/lib/languages/plaintext'
import python from 'highlight.js/lib/languages/python'
import yaml from 'highlight.js/lib/languages/yaml'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { LAB_SOURCES, layaInputs } from './knowledge.js'

hljs.registerLanguage('dockerfile', dockerfile)
hljs.registerLanguage('python', python)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('toml', ini)
hljs.registerLanguage('text', plaintext)

const snapshot = ref({ episodes: [], metadata: {} })
const connected = ref(false)
const error = ref('')
const showResetDialog = ref(false)
const resetting = ref(false)
const resetError = ref('')
const resetStartedAt = ref(null)
const sourceFiles = ref([])
const selectedSource = ref(null)
const sourceError = ref('')
const selectedSavestate = ref('')
const savestateSelectionInitialized = ref(false)
const rolloutVideos = ref([])
let timer
let mediaTimer

const highlightedSource = computed(() => {
  if (!selectedSource.value?.content) return ''
  const language = selectedSource.value.language
  if (language && hljs.getLanguage(language)) {
    return hljs.highlight(selectedSource.value.content, { language }).value
  }
  return hljs.highlightAuto(selectedSource.value.content).value
})

const load = async () => {
  try {
    const response = await fetch('/api/snapshot', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    snapshot.value = payload
    if (!savestateSelectionInitialized.value) {
      selectedSavestate.value = payload.metadata?.run?.savestate || ''
      savestateSelectionInitialized.value = true
    }
    if (resetting.value && resetStartedAt.value && payload.started_at !== resetStartedAt.value) {
      resetting.value = false
      resetStartedAt.value = null
    }
    connected.value = true
    error.value = ''
  } catch (reason) {
    connected.value = false
    error.value = reason.message
  }
}

const loadSource = async path => {
  sourceError.value = ''
  try {
    const response = await fetch(`/api/source?path=${encodeURIComponent(path)}`, { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    selectedSource.value = await response.json()
  } catch (reason) {
    sourceError.value = reason.message
  }
}

const loadSources = async () => {
  try {
    const response = await fetch('/api/sources', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const payload = await response.json()
    sourceFiles.value = payload.files || []
    if (sourceFiles.value.length) await loadSource(sourceFiles.value[0].path)
  } catch (reason) {
    sourceError.value = reason.message
  }
}

const reports = ref([])
const selectedReport = ref(null)
const reportError = ref('')
const renderedReport = computed(() => selectedReport.value ? DOMPurify.sanitize(marked.parse(selectedReport.value.content)) : '')

const loadReport = async name => {
  reportError.value = ''
  try {
    const response = await fetch(`/api/report?name=${encodeURIComponent(name)}`, { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    selectedReport.value = await response.json()
  } catch (reason) {
    reportError.value = reason.message
  }
}

const loadReports = async () => {
  try {
    const response = await fetch('/api/reports', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    reports.value = (await response.json()).reports
    if (reports.value.length && !selectedReport.value) await loadReport(reports.value[0].name)
  } catch (reason) {
    reportError.value = reason.message
  }
}

const loadRolloutVideos = async () => {
  try {
    const response = await fetch('/api/rollout-videos', { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    rolloutVideos.value = (await response.json()).videos || []
  } catch {
    rolloutVideos.value = []
  }
}

onMounted(() => {
  load(); loadSources(); loadRolloutVideos(); loadReports()
  timer = window.setInterval(load, 1500)
  mediaTimer = window.setInterval(() => { loadRolloutVideos(); loadReports() }, 5000)
})
onBeforeUnmount(() => { window.clearInterval(timer); window.clearInterval(mediaTimer) })

const summary = computed(() => snapshot.value.summary || {})
const savestateSummaries = computed(() => summary.value.by_savestate || {})
const savestateCurriculum = computed(() => snapshot.value.metadata?.savestate_curriculum || {})
const curriculumRows = computed(() => Object.entries(savestateCurriculum.value).map(([state, value]) => {
  const curriculum = value && typeof value === 'object' ? value : {}
  const wins = Math.max(0, Number(curriculum.wins) || 0)
  const target = Math.max(1, Number(curriculum.win_target))
  return {
    state,
    wins,
    target,
    mastered: Boolean(curriculum.mastered) || wins >= target,
    active: Boolean(curriculum.active),
  }
}))
const currentCurriculumState = computed(() => curriculumRows.value.find(row => row.active)
  || curriculumRows.value.find(row => !row.mastered)
  || null)
const curriculumComplete = computed(() => curriculumRows.value.length > 0
  && curriculumRows.value.every(row => row.mastered))
const masteredStateCount = computed(() => curriculumRows.value.filter(row => row.mastered).length)
const currentStateProgress = computed(() => currentCurriculumState.value
  ? Math.min(100, currentCurriculumState.value.wins / currentCurriculumState.value.target * 100)
  : 0)
const availableSavestates = computed(() => Object.keys(savestateSummaries.value).filter(Boolean)
  .sort((left, right) => left.localeCompare(right)))
const activeSummary = computed(() => selectedSavestate.value
  ? savestateSummaries.value[selectedSavestate.value] || {}
  : summary.value)
const activeSavestateLabel = computed(() => selectedSavestate.value || 'All savestates')
const initialSavestate = computed(() => selectedSavestate.value || run.value.savestate || '')
const videosForSelectedSavestate = computed(() => rolloutVideos.value.filter(video =>
  !selectedSavestate.value || !video.savestate || video.savestate === selectedSavestate.value))
const bestVideo = videos => videos.reduce((best, candidate) =>
  best == null || Number(candidate.score) > Number(best.score) ? candidate : best, null)
const bestCurriculumVideos = computed(() => curriculumRows.value.map(row => ({
  ...row,
  video: bestVideo(videosForSelectedSavestate.value.filter(video => video.curriculum === row.state)),
})))
const bestFullRunVideo = computed(() => bestVideo(rolloutVideos.value.filter(video =>
  video.started_from_initial_savestate === true
  && video.episode_start_state === initialSavestate.value
  && (!selectedSavestate.value || !video.savestate || video.savestate === selectedSavestate.value))))
const summarizedEpisodes = computed(() => Number(activeSummary.value.full_run_episodes) || 0)
const best = computed(() => activeSummary.value.full_run_best_fitness ?? null)
const wins = computed(() => Number(activeSummary.value.full_run_wins) || 0)
const winRate = computed(() => summarizedEpisodes.value ? wins.value / summarizedEpisodes.value * 100 : 0)
const summarizedAvgDuration = computed(() => {
  const timed = Number(activeSummary.value.full_run_timed_episodes) || 0
  return timed ? Number(activeSummary.value.full_run_duration_seconds_total) / timed : null
})
const latestTrainingState = computed(() => activeSummary.value.latest_training_state || summary.value.latest_training_state)
const latestDuration = computed(() => activeSummary.value.latest_full_run_duration_seconds ?? null)
const model = computed(() => snapshot.value.metadata?.model || {})
const laya = computed(() => model.value.laya || {})
const knowledge = computed(() => snapshot.value.metadata?.knowledge)
const memory = computed(() => snapshot.value.metadata?.memory || { state: null, fields: [] })
const memoryValue = value => Array.isArray(value) ? value.join(' · ') : String(value)
const inputs = computed(() => layaInputs(knowledge.value, laya.value))
const stateSources = state => [state.seeded ? 'seeded from power-on' : 'no seed', `${state.demonstrations} demonstrated moves`].join(' · ')
const environment = computed(() => snapshot.value.metadata?.environment || {})
const run = computed(() => snapshot.value.metadata?.run || {})
const server = computed(() => snapshot.value.server || {})
const versionLabel = computed(() => server.value.version === 'DEVELOPMENT'
  ? 'DEVELOPMENT'
  : server.value.version ? `v${server.value.version}` : '—')
const control = computed(() => snapshot.value.control || {})

const resetModel = async () => {
  resetting.value = true
  resetStartedAt.value = snapshot.value.started_at
  resetError.value = ''
  try {
    const response = await fetch('/api/model/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': control.value.csrf_token },
      body: JSON.stringify({ game: run.value.game }),
    })
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`)
    showResetDialog.value = false
  } catch (reason) {
    resetting.value = false
    resetError.value = reason.message
  }
}

const fmt = (value, digits = 0) => value == null ? '—' : Intl.NumberFormat('en', { maximumFractionDigits: digits }).format(value)
const duration = value => {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  const seconds = Math.max(0, Math.round(Number(value)))
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
}
const display = value => {
  if (value == null || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (Array.isArray(value)) return value.join('\n')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
const entries = object => Object.entries(object || {}).filter(([, value]) => value != null && value !== 'None')
const label = key => key.replaceAll('_', ' ')
</script>

<template>
  <main>
    <header class="topbar">
      <div class="brand">
        <div class="mark"><span></span><span></span><span></span></div>
        <div><p class="eyebrow">DATENWISSENSCHAFTEN · {{ versionLabel }}</p><h1>Training Observatory</h1></div>
      </div>
      <div class="run-summary">
        <span>{{ run.game || 'Awaiting run' }}</span>
        <span class="separator">/</span>
        <span>{{ latestTrainingState || run.savestate || 'no state' }}</span>
        <span class="endpoint">{{ server.bind_address || '—' }}</span>
        <a class="stream-link" href="/stream" target="_blank">Stream view</a>
        <span :class="['connection', { offline: !connected }]"><i></i>{{ connected ? 'Live' : 'Disconnected' }}</span>
      </div>
    </header>

    <section class="controls panel">
      <div><p class="eyebrow">OBSERVATION WINDOW</p><strong>Episode telemetry</strong></div>
      <label>Savestate
        <select v-model="selectedSavestate">
          <option value="">All savestates</option>
          <option v-for="savestate in availableSavestates" :key="savestate" :value="savestate">{{ savestate }}</option>
        </select>
      </label>
      <button class="reset-button" :disabled="!control.restart_supported || control.reset_pending || resetting" @click="showResetDialog = true">
        {{ control.reset_pending || resetting ? 'Restarting…' : 'Delete model' }}
      </button>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section class="kpis">
      <article class="panel metric"><p>Best fitness</p><strong class="mint">{{ fmt(best, 2) }}</strong><small>Runs from {{ activeSavestateLabel }}</small></article>
      <article class="panel metric"><p>Win rate</p><strong>{{ fmt(winRate, 1) }}<em>%</em></strong><small>{{ wins }} successful / {{ summarizedEpisodes }} episodes</small></article>
      <article class="panel metric"><p>Avg training time</p><strong>{{ duration(summarizedAvgDuration) }}</strong><small>{{ duration(latestDuration) }} latest episode</small></article>
      <article class="panel metric"><p>Full runs</p><strong>{{ fmt(summarizedEpisodes) }}</strong><small>Started from {{ activeSavestateLabel }}</small></article>
    </section>

    <section v-if="curriculumRows.length && !curriculumComplete" class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">STATE CURRICULUM</p><h2>Current training state</h2><p>Only the state currently being learned and overall sequence progress are shown.</p></div>
        <span>{{ masteredStateCount }} / {{ curriculumRows.length }} mastered</span>
      </div>
      <article v-if="currentCurriculumState" class="panel curriculum-progress">
        <div class="card-heading">
          <div><p class="eyebrow">TRAINING NOW</p><h2>{{ currentCurriculumState.state }}</h2></div>
          <span class="chip">{{ fmt(currentCurriculumState.wins) }} / {{ fmt(currentCurriculumState.target) }} beaten</span>
        </div>
        <div class="progress-track" role="progressbar" :aria-valuenow="currentCurriculumState.wins" aria-valuemin="0" :aria-valuemax="currentCurriculumState.target">
          <span :style="{ width: `${currentStateProgress}%` }"></span>
        </div>
        <div class="state-sequence" aria-label="Curriculum state sequence">
          <span
            v-for="row in curriculumRows"
            :key="row.state"
            :class="{ mastered: row.mastered, current: row.state === currentCurriculumState.state }"
          >{{ row.state }}</span>
        </div>
      </article>
    </section>

    <section v-if="curriculumRows.length && !curriculumComplete" class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">SAVESTATE DEBUG VIDEOS</p><h2>Best rollout from every training state</h2><p>Only the highest-fitness checkpoint rollout remains selected while the state curriculum is being learned.</p></div>
        <span>{{ bestCurriculumVideos.filter(row => row.video).length }} / {{ curriculumRows.length }} recorded</span>
      </div>
      <div class="state-video-grid">
        <article v-for="row in bestCurriculumVideos" :key="row.state" class="panel state-video-card">
          <div class="state-video-heading">
            <div><p class="eyebrow">{{ row.mastered ? 'MASTERED' : row.active ? 'TRAINING NOW' : 'WAITING' }}</p><strong>{{ row.state }}</strong></div>
            <span v-if="row.video">Rollout {{ fmt(row.video.rollout) }}</span>
          </div>
          <template v-if="row.video">
            <video controls preload="metadata" :src="`/api/rollout-video?path=${encodeURIComponent(row.video.path)}`"></video>
            <div class="state-video-meta"><span>Score {{ fmt(row.video.score, 2) }}</span><span>{{ row.video.curriculum_succeeded ? 'Beaten' : 'Not completed' }}</span></div>
          </template>
          <p v-else class="state-video-empty">No completed episode has been recorded from this state yet.</p>
        </article>
      </div>
    </section>

    <section v-if="curriculumComplete && bestFullRunVideo" class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">INITIAL SAVESTATE</p><h2>Best full-run rollout</h2><p>The highest-fitness run recorded from {{ activeSavestateLabel }}, without an automatic checkpoint.</p></div>
        <span>Rollout {{ fmt(bestFullRunVideo.rollout) }}</span>
      </div>
      <article class="panel full-run-video">
        <div class="state-video-heading">
          <strong>Score {{ fmt(bestFullRunVideo.score, 2) }}</strong>
          <span>{{ bestFullRunVideo.won ? 'Won' : 'Not completed' }}</span>
        </div>
        <video controls preload="metadata" :src="`/api/rollout-video?path=${encodeURIComponent(bestFullRunVideo.path)}`"></video>
      </article>
    </section>

    <section class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">KNOWLEDGE</p><h2>Where Laya's knowledge comes from</h2><p>The lab turns published sources and its own emulator measurements into states, questions, rewards, start points and demonstrations. Laya only learns from those, never from a website.</p></div>
      </div>
      <div class="details-grid system-grid two-column">
      <article class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">THE LAB</p><h2>Sources, best first</h2><p class="card-description">A higher source wins when two disagree; the emulator wins over all of them.</p></div><span class="chip">{{ LAB_SOURCES.length }} kinds</span></div>
        <dl><template v-for="(source, index) in LAB_SOURCES" :key="source.name"><dt>{{ index + 1 }}. {{ source.name }}</dt><dd>{{ source.detail }}</dd></template></dl>
      </article>
      <article class="panel detail-card full-height">
        <div class="card-heading"><div><p class="eyebrow">LAYA</p><h2>What Laya learns from</h2><p class="card-description">Imitation of the demonstrations stops once a state is mastered; then only the reward counts.</p></div><span class="chip" :class="{ muted: !inputs.length }">{{ inputs.length ? 'Live' : 'Loading' }}</span></div>
        <dl>
          <template v-for="input in inputs" :key="input.name"><dt>{{ input.name }}</dt><dd>{{ input.value }}</dd></template>
          <template v-for="state in knowledge?.states || []" :key="state.name"><dt>{{ state.name }}</dt><dd>{{ stateSources(state) }}</dd></template>
        </dl>
      </article>
      </div>
    </section>

    <section class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">GAME MEMORY</p><h2>Every RAM value the game package reads</h2><p>Live values from the running attempt, refreshed every ten decisions{{ memory.state ? ` · state ${memory.state}` : '' }}.</p></div>
      </div>
      <article class="panel detail-card full-height">
        <div class="card-heading"><div><p class="eyebrow">GAMERAM</p><h2>{{ memory.fields.length }} fields</h2><p class="card-description">Name, NES address and the value Laya's state machine sees right now.</p></div><span class="chip" :class="{ muted: !memory.fields.length }">{{ memory.fields.length ? 'Live' : 'Waiting' }}</span></div>
        <dl class="memory-list"><template v-for="field in memory.fields" :key="field.name"><dt>{{ field.name }} <span class="memory-address">{{ field.address }}{{ field.length > 1 ? ` +${field.length}` : '' }}</span></dt><dd>{{ memoryValue(field.value) }}</dd></template></dl>
      </article>
    </section>

    <section class="observatory-section">
      <div class="section-heading">
        <div><p class="eyebrow">TRAINING ENGINE</p><h2>Runtime and learning system</h2><p>The game {{ server.persona }} plays and the Laya model making every decision.</p></div>
      </div>
      <div class="details-grid system-grid two-column">
      <article class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">GAME</p><h2>Environment</h2><p class="card-description">Emulator, state machine, and the actions Laya can choose.</p></div><span class="chip">1 env</span></div>
        <dl><template v-for="([key, value]) in entries(environment)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      <article class="panel detail-card">
        <div class="card-heading"><div><p class="eyebrow">DECISION MODEL</p><h2>{{ model.display_name || 'Laya' }}</h2><p class="card-description">{{ model.description || 'Laya details appear when training starts.' }}</p></div><span class="chip" :class="{ muted: !entries(laya).length }">{{ entries(laya).length ? 'Active' : 'Loading' }}</span></div>
        <dl><template v-for="([key, value]) in entries(laya)" :key="key"><dt>{{ label(key) }}</dt><dd>{{ display(value) }}</dd></template></dl>
      </article>
      </div>
    </section>

    <section class="panel source-browser">
      <div class="source-browser-heading">
        <div><p class="eyebrow">DAILY AGENT</p><h2>Reports</h2></div>
        <span class="chip" :class="{ muted: !reports.length }">{{ reports.length }} reports</span>
      </div>
      <div v-if="reports.length" class="source-browser-body">
        <nav class="source-files" aria-label="Daily reports">
          <button
            v-for="report in reports"
            :key="report.name"
            :class="{ active: selectedReport?.name === report.name }"
            @click="loadReport(report.name)"
          >
            <strong>{{ report.name.replace('.md', '') }}</strong><small>{{ fmt(report.size) }} B</small>
          </button>
        </nav>
        <article class="source-viewer report-viewer">
          <header v-if="selectedReport"><strong>{{ selectedReport.name }}</strong><span>markdown</span></header>
          <div v-if="selectedReport" class="report-body" v-html="renderedReport"></div>
          <p v-else class="placeholder">Choose a report to read it.</p>
        </article>
      </div>
      <p v-else-if="reportError" class="error">Reports could not be loaded: {{ reportError }}</p>
      <p v-else class="placeholder">No daily report yet. The first one appears after the next daily run.</p>
    </section>

    <section class="panel source-browser">
      <div class="source-browser-heading">
        <div><p class="eyebrow">GENERATED PROJECT</p><h2>Files and source code</h2></div>
        <span class="chip" :class="{ muted: !sourceFiles.length }">{{ sourceFiles.length }} files</span>
      </div>
      <div v-if="sourceFiles.length" class="source-browser-body">
        <nav class="source-files" aria-label="Generated files">
          <button
            v-for="file in sourceFiles"
            :key="file.path"
            :class="{ active: selectedSource?.path === file.path }"
            @click="loadSource(file.path)"
          >
            <strong>{{ file.path }}</strong><small>{{ file.language }} · {{ fmt(file.size) }} B</small>
          </button>
        </nav>
        <article class="source-viewer">
          <header v-if="selectedSource"><strong>{{ selectedSource.path }}</strong><span>{{ selectedSource.language }}</span></header>
          <pre v-if="selectedSource"><code class="hljs" v-html="highlightedSource"></code></pre>
          <p v-else class="placeholder">Choose a generated file to inspect it.</p>
        </article>
      </div>
      <p v-else-if="sourceError" class="error">Generated files could not be loaded: {{ sourceError }}</p>
      <p v-else class="placeholder">No generated runner files were found.</p>
    </section>

    <footer>Local telemetry · refreshes every 1.5 seconds · {{ summarizedEpisodes }} full runs from {{ activeSavestateLabel }}</footer>

    <div v-if="showResetDialog" class="modal-backdrop" @click.self="showResetDialog = false">
      <section class="reset-dialog panel" role="dialog" aria-modal="true" aria-labelledby="reset-title">
        <p class="eyebrow danger-text">DESTRUCTIVE ACTION</p>
        <h2 id="reset-title">Delete {{ run.game }} model?</h2>
        <p>All models, recordings, cache data, persisted runtime states, and training history for this runner will be deleted. The active training run will restart from its configured savestate.</p>
        <p v-if="resetError" class="error">{{ resetError }}</p>
        <div class="dialog-actions">
          <button class="cancel-button" :disabled="resetting" @click="showResetDialog = false">Cancel</button>
          <button class="confirm-reset" :disabled="resetting" @click="resetModel">{{ resetting ? 'Requesting…' : 'Delete and restart' }}</button>
        </div>
      </section>
    </div>
  </main>
</template>
