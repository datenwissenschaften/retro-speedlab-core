<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ProgressPath from './ProgressPath.vue'
import { createObsControl } from './obsControl.js'
import { createReplayPlayer } from './replayPlayer.js'
import SpotlightPanel from './SpotlightPanel.vue'
import StoryTicker from './StoryTicker.vue'
import './stream.css'

const STAGE_WIDTH = 1280
const STAGE_HEIGHT = 720
const SNAPSHOT_INTERVAL_MS = 1500
const TOAST_MS = 4500
const TICKER_SIZE = 3
const TICKER_MS = 8000
const MAX_TICKER_BACKLOG = 6
const SITE_URL = 'https://www.retrospeedlab.com'
const SITE_LABEL = 'www.retrospeedlab.com'
const RELOAD_DEADLINE_MS = 90000

const live = ref({})
const screen = ref(null)
const replayEpisode = ref(null)
const waiting = ref(true)
const replayProgress = ref(0)
const snapshot = ref({ metadata: {}, summary: {} })
const connected = ref(false)
const scale = ref(1)
const toast = ref(null)
const changedFields = ref(new Set())
const tickerEvents = ref([])
const tickerQueue = []
let lastStep = null
let eventKey = 0
let snapshotTimer
let toastTimer
let reloadTimer
let loadedRelease = null
let reloadPending = false

const fmt = (value, digits = 0) => value == null ? '—' : Intl.NumberFormat('en', { maximumFractionDigits: digits }).format(value)
const percent = value => `${Math.round(value * 100)}%`
const label = name => name.replaceAll('_', ' ')
const readable = value => {
  if (Array.isArray(value)) return value.join(' · ')
  if (value && typeof value === 'object') {
    if (value.visible) return [value.direction, value.distance == null ? null : `${value.distance}px`].filter(Boolean).join(' · ')
    return value.remembered ? `remembered · ${value.move}` : 'not visible'
  }
  return String(value)
}
const announce = (title, detail) => {
  toast.value = { title, detail, key: Date.now() }
  window.clearTimeout(toastTimer)
  toastTimer = window.setTimeout(() => { toast.value = null }, TOAST_MS)
}

const showQueued = () => {
  while (tickerEvents.value.length < TICKER_SIZE && tickerQueue.length) {
    const entry = tickerQueue.shift()
    tickerEvents.value = [...tickerEvents.value, entry]
    window.setTimeout(() => {
      tickerEvents.value = tickerEvents.value.filter(other => other.key !== entry.key)
      showQueued()
    }, TICKER_MS)
  }
}
const tell = item => {
  tickerQueue.push({ ...item, key: eventKey += 1 })
  tickerQueue.splice(0, Math.max(0, tickerQueue.length - MAX_TICKER_BACKLOG))
  showQueued()
  if (item.kind === 'milestone') announce(item.text, item.detail)
}
const tellStep = status => {
  if (status.step === lastStep) return
  lastStep = status.step
  status.events.forEach(tell)
}

const drawFrame = frame => {
  const canvas = screen.value
  if (canvas.width !== frame.bitmap.width) {
    canvas.width = frame.bitmap.width
    canvas.height = frame.bitmap.height
  }
  canvas.getContext('2d').drawImage(frame.bitmap, 0, 0)
  live.value = frame.status
  replayProgress.value = frame.progress
  frame.passed.forEach(tellStep)
}
const reloadIfPending = () => { if (reloadPending) window.location.reload() }
const player = createReplayPlayer({
  onFrame: drawFrame,
  onEpisode: episode => {
    replayEpisode.value = episode
    waiting.value = false
  },
  onWaiting: () => {
    waiting.value = true
    reloadIfPending()
  },
  onEpisodeEnd: reloadIfPending,
  onSummary: () => {},
  onConnection: online => { connected.value = online },
})
const obsControl = createObsControl()
const loadSnapshot = async () => {
  try {
    const response = await fetch('/api/snapshot', { cache: 'no-store' })
    if (response.ok) snapshot.value = await response.json()
  } catch {
    connected.value = false
  }
}
const fit = () => { scale.value = Math.min(window.innerWidth / STAGE_WIDTH, window.innerHeight / STAGE_HEIGHT) }
const stageStyle = computed(() => ({
  transform: `translate(${(window.innerWidth - STAGE_WIDTH * scale.value) / 2}px, ${(window.innerHeight - STAGE_HEIGHT * scale.value) / 2}px) scale(${scale.value})`,
}))

onMounted(() => {
  fit(); loadSnapshot(); player.start(); obsControl.start()
  window.addEventListener('resize', fit)
  snapshotTimer = window.setInterval(loadSnapshot, SNAPSHOT_INTERVAL_MS)
})
onBeforeUnmount(() => {
  player.stop(); obsControl.stop()
  window.removeEventListener('resize', fit)
  window.clearInterval(snapshotTimer); window.clearTimeout(toastTimer); window.clearTimeout(reloadTimer)
})

const release = computed(() => snapshot.value.server?.release || null)
const run = computed(() => snapshot.value.metadata?.run || {})
const summary = computed(() => snapshot.value.summary || {})
const story = computed(() => snapshot.value.metadata?.story || { phases: [], danger: [] })
const curriculum = computed(() => snapshot.value.metadata?.savestate_curriculum || {})
const phase = computed(() => live.value.training_state || story.value.phases[0]?.name || '')
const areasReached = computed(() => story.value.phases.filter(item => item.reached).length)
const probabilities = computed(() => Object.entries(live.value.probabilities || {}))
const confidence = computed(() => Math.max(0, ...probabilities.value.map(([, p]) => p)))
const isSighting = value => Boolean(value) && typeof value === 'object' && 'visible' in value
const sightings = computed(() => Object.entries(live.value.ram || {}).filter(([, value]) => isSighting(value)))
const ramState = computed(() => Object.entries(live.value.ram || {})
  .filter(([name, value]) => !isSighting(value) && name !== 'snake_visible'))

watch(release, current => {
  if (!current) return
  if (loadedRelease === null) {
    loadedRelease = current
    return
  }
  if (current === loadedRelease || reloadPending) return
  reloadPending = true
  if (waiting.value) window.location.reload()
  reloadTimer = window.setTimeout(() => window.location.reload(), RELOAD_DEADLINE_MS)
})

watch(() => live.value.ram, (current, previous) => {
  if (!current || !previous) return
  changedFields.value = new Set(Object.keys(current).filter(key => JSON.stringify(current[key]) !== JSON.stringify(previous[key])))
})
</script>

<template>
  <div class="stream-view">
    <div class="stream-stage" :style="stageStyle">
      <div class="stream-left-rail">
        <aside class="run-info-panel">
          <div class="run-info-brand">
            <img class="run-info-logo" src="/logo.png" alt="Retro Speedlab" />
            <span class="run-info-kicker">{{ connected ? 'Live' : 'Offline' }}</span>
          </div>
          <div class="run-info-medal">
            <span class="run-info-medal-icon">🧠</span>
            <span>
              <strong class="run-info-medal-title">Attempt {{ replayEpisode ? `#${replayEpisode.id}` : '—' }}</strong>
              <span class="replay-track"><span :style="{ width: percent(replayProgress) }"></span></span>
            </span>
          </div>
          <dl class="run-info-grid">
            <div class="run-info-row"><dt>Game</dt><dd>{{ run.game || 'Waiting' }}</dd></div>
            <div class="run-info-row"><dt>Level</dt><dd>{{ run.savestate || '—' }}</dd></div>
            <div class="run-info-row"><dt>Model</dt><dd>Laya {{ release || '—' }}</dd></div>
          </dl>
        </aside>

        <section class="sight-panel">
          <span class="sight-title">In sight</span>
          <div v-for="[name, value] in sightings" :key="name" :class="['sighting', { seen: value.visible }]">
            <strong>{{ label(name) }}</strong>
            <span>{{ readable(value) }}</span>
          </div>
          <span v-if="!sightings.length" class="game-cover-loading">Waiting for the first attempt</span>
        </section>

        <SpotlightPanel :danger="story.danger" />
      </div>

      <div class="stream-screen">
        <canvas ref="screen" class="stream-video" aria-label="Replayed gameplay"></canvas>
        <StoryTicker :events="tickerEvents" />
        <Transition name="fade">
          <div v-if="waiting" class="stream-waiting">
            <span class="stream-waiting-kicker">Next replay loading</span>
            <strong class="stream-waiting-title">Laya trains<span class="stream-waiting-dots"><i>.</i><i>.</i><i>.</i></span></strong>
            <span class="stream-waiting-copy">The next attempt appears here as soon as it is finished.</span>
            <dl class="stream-waiting-stats">
              <div><dt>Attempts</dt><dd>{{ fmt(summary.episodes) }}</dd></div>
              <div><dt>Best score</dt><dd>{{ fmt(summary.best_fitness, 1) }}</dd></div>
              <div><dt>Areas reached</dt><dd>{{ areasReached }} / {{ story.phases.length }}</dd></div>
            </dl>
          </div>
        </Transition>
      </div>

      <aside class="stream-ad-panel brain-panel">
        <strong class="stream-ad-title">Thinking</strong>
        <span class="stream-ad-copy">“{{ live.question || 'Waiting for the first finished attempt…' }}”</span>
        <ul class="brain-options">
          <li v-for="[name, probability] in probabilities" :key="name" :class="{ chosen: name === live.action }">
            <div class="brain-option-label"><strong>{{ name }}</strong><b>{{ percent(probability) }}</b></div>
            <div class="brain-bar"><span :style="{ width: percent(probability) }"></span></div>
          </li>
        </ul>
        <span class="stream-ad-url">{{ percent(confidence) }} sure · {{ probabilities.length }} options</span>
        <dl class="brain-ram">
          <template v-for="[name, value] in ramState" :key="name">
            <dt>{{ label(name) }}</dt><dd :class="{ flash: changedFields.has(name) }">{{ readable(value) }}</dd>
          </template>
        </dl>
      </aside>

      <div class="stream-bottom">
        <ProgressPath :phases="story.phases" :current="phase" :curriculum="curriculum" />
        <a class="site-card" :href="SITE_URL" target="_blank" rel="noopener noreferrer">
          <span class="sight-title">Train your own runner</span>
          <strong class="site-url">{{ SITE_LABEL }}</strong>
          <span class="site-score">Best score {{ fmt(summary.best_fitness, 1) }} · {{ fmt(summary.episodes) }} attempts</span>
        </a>
      </div>

      <Transition name="toast">
        <div v-if="toast" :key="toast.key" class="stream-toast"><strong>{{ toast.title }}</strong><span>{{ toast.detail }}</span></div>
      </Transition>
    </div>
  </div>
</template>
