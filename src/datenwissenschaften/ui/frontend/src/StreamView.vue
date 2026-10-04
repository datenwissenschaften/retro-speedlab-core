<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ProgressPath from './ProgressPath.vue'
import { createArrivalTracker, holdsBest, inProgressLine, recentAttempts, status } from './attempts.js'
import { fmt, gameTitle, percent, words } from './naming.js'
import { createObsControl, STREAM_TIME_ZONE } from './obsControl.js'
import { createReplayPlayer } from './replayPlayer.js'
import { elapsed } from './runtime.js'
import SpotlightPanel from './SpotlightPanel.vue'
import './stream.css'

const STAGE_WIDTH = 1280
const STAGE_HEIGHT = 720
const SNAPSHOT_INTERVAL_MS = 1500
const RELOAD_DEADLINE_MS = 90000
const RELOAD_SETTLE_MS = 15000
const RELOAD_RETRY_MS = 5000
const CLOCK_INTERVAL_MS = 1000
const RECENT_ATTEMPTS = 8
const clock = new Intl.DateTimeFormat('en-GB', { timeZone: STREAM_TIME_ZONE, dateStyle: 'medium', timeStyle: 'medium' })

const live = ref({})
const screen = ref(null)
const replayEpisode = ref(null)
const latestEpisode = ref(null)
const recentScores = ref([])
const inProgress = ref(null)
const bestRefresh = ref(0)
const waiting = ref(true)
const replayProgress = ref(0)
const snapshot = ref({ metadata: {}, summary: {} })
const connected = ref(false)
const scale = ref(1)
const now = ref(new Date())
const changedFields = ref(new Set())
let snapshotTimer
let clockTimer
const arrived = createArrivalTracker()
let reloadTimer
let loadedRelease = null
let reloadPending = false
let reloading = false

const label = name => name.replaceAll('_', ' ')
const readable = value => {
  if (Array.isArray(value)) return value.join(' · ')
  if (value && typeof value === 'object') {
    if (value.visible) return [value.direction, value.distance == null ? null : `${value.distance}px`].filter(Boolean).join(' · ')
    return value.remembered ? `remembered · ${value.move}` : 'not visible'
  }
  return String(value)
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
}
const pause = milliseconds => new Promise(resolve => { window.setTimeout(resolve, milliseconds) })
const pageServed = async () => {
  try {
    const response = await fetch(window.location.href, { cache: 'no-store' })
    return response.ok && (await response.text()).includes('id="app"')
  } catch {
    return false
  }
}
const reload = async () => {
  if (reloading) return
  reloading = true
  window.clearTimeout(reloadTimer)
  await pause(RELOAD_SETTLE_MS)
  while (!(await pageServed())) await pause(RELOAD_RETRY_MS)
  window.location.reload()
}
const reloadIfPending = () => { if (reloadPending) reload() }
const player = createReplayPlayer({
  onFrame: drawFrame,
  onEpisode: (episode, replay, generation) => {
    replayEpisode.value = episode
    waiting.value = false
    if (arrived(generation, episode, replay) && (episode.result.won || episode.result.new_best)) bestRefresh.value += 1
  },
  onWaiting: () => {
    waiting.value = true
    reloadIfPending()
  },
  onEpisodeEnd: reloadIfPending,
  onLatest: (episode, summary, running) => {
    latestEpisode.value = episode
    recentScores.value = summary.recent_scores || []
    inProgress.value = running
  },
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

const twitch = computed(() => snapshot.value.server?.twitch === true)
let streaming = false
watch(twitch, enabled => {
  if (!enabled || streaming) return
  streaming = true
  player.start(); obsControl.start()
})

onMounted(() => {
  fit(); loadSnapshot()
  window.addEventListener('resize', fit)
  snapshotTimer = window.setInterval(loadSnapshot, SNAPSHOT_INTERVAL_MS)
  clockTimer = window.setInterval(() => { now.value = new Date() }, CLOCK_INTERVAL_MS)
})
onBeforeUnmount(() => {
  player.stop(); obsControl.stop()
  window.removeEventListener('resize', fit)
  window.clearInterval(snapshotTimer); window.clearInterval(clockTimer); window.clearTimeout(reloadTimer)
})

const release = computed(() => snapshot.value.server?.release || null)
const run = computed(() => snapshot.value.metadata?.run || {})
const level = computed(() => live.value.level || run.value.savestate || '')
const levelTitle = computed(() => words(level.value))
const summary = computed(() => snapshot.value.summary?.by_savestate?.[level.value] || {})
const story = computed(() => snapshot.value.metadata?.stories?.[level.value] || { phases: [], danger: [] })
const levelBest = savestate => snapshot.value.summary?.by_savestate?.[savestate]?.best_fitness ?? null
const recent = computed(() => recentAttempts(latestEpisode.value, recentScores.value, levelBest(latestEpisode.value?.result.level), RECENT_ATTEMPTS))
const replayStatus = computed(() => status(connected.value, replayEpisode.value))
const progressLine = computed(() => connected.value ? inProgressLine(replayEpisode.value, inProgress.value) : null)
const replayIsBest = computed(() => replayEpisode.value !== null && holdsBest(replayEpisode.value, levelBest(replayEpisode.value.result.level)))
const learningFor = computed(() => snapshot.value.started_at ? elapsed(snapshot.value.started_at, now.value) : '—')
const agentName = computed(() => snapshot.value.metadata?.model?.display_name || '—')
const areasReached = computed(() => story.value.phases.filter(item => item.reached).length)
const probabilities = computed(() => Object.entries(live.value.probabilities || {}))
const confidence = computed(() => Math.max(0, ...probabilities.value.map(([, p]) => p)))
const isSighting = value => Boolean(value) && typeof value === 'object' && 'visible' in value
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
  if (waiting.value) reload()
  reloadTimer = window.setTimeout(reload, RELOAD_DEADLINE_MS)
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
            <span class="run-info-kicker">{{ replayStatus }}</span>
          </div>
          <div class="run-info-medal">
            <span class="run-info-medal-icon">🧠</span>
            <span>
              <strong class="run-info-medal-title">Attempt {{ replayEpisode ? `#${replayEpisode.result.attempt} · ${words(replayEpisode.result.level)}` : '—' }}</strong>
              <span v-if="progressLine" class="replay-badge">{{ progressLine }}</span>
              <span v-if="replayIsBest" class="replay-badge best">★ Best so far</span>
              <span class="replay-track"><span :style="{ width: percent(replayProgress) }"></span></span>
            </span>
          </div>
          <dl class="run-info-grid">
            <div class="run-info-row"><dt>Game</dt><dd>{{ run.game ? gameTitle(run.game) : 'Waiting' }}</dd></div>
            <div class="run-info-row"><dt>Level</dt><dd>{{ levelTitle || '—' }}</dd></div>
            <div class="run-info-row"><dt>Agent</dt><dd>{{ agentName }}</dd></div>
          </dl>
        </aside>

        <SpotlightPanel :danger="story.danger" :failures="story.failures || 0" :level="level" :recent="recent" :refresh="bestRefresh" />
      </div>

      <div class="stream-screen">
        <canvas ref="screen" class="stream-video" aria-label="Replayed gameplay"></canvas>
        <Transition name="fade">
          <div v-if="snapshot.server && !twitch" class="stream-waiting">
            <span class="stream-waiting-kicker">Stream off</span>
            <strong class="stream-waiting-title">Twitch is disabled</strong>
            <span class="stream-waiting-copy">Set twitch.enabled to true in config.yaml to stream the experiment.</span>
          </div>
          <div v-else-if="waiting" class="stream-waiting">
            <span class="stream-waiting-kicker">Next replay loading</span>
            <strong class="stream-waiting-title">{{ agentName }} keeps exploring<span class="stream-waiting-dots"><i>.</i><i>.</i><i>.</i></span></strong>
            <span class="stream-waiting-copy">The next attempt appears here as soon as it is finished.</span>
            <dl class="stream-waiting-stats">
              <div><dt>Attempts</dt><dd>{{ fmt(summary.episodes) }}</dd></div>
              <div><dt>Best reward</dt><dd>{{ fmt(summary.best_fitness, 1) }}</dd></div>
              <div><dt>Areas reached</dt><dd>{{ areasReached }} / {{ story.phases.length }}</dd></div>
            </dl>
          </div>
        </Transition>
      </div>

      <aside class="stream-ad-panel brain-panel">
        <strong class="stream-ad-title">Next move</strong>
        <span class="stream-ad-copy">{{ probabilities.length ? `Choosing from ${probabilities.length} actions` : 'Waiting for the first finished attempt…' }}</span>
        <ul class="brain-options">
          <li v-for="[name, probability] in probabilities" :key="name" :class="{ chosen: name === live.action }">
            <div class="brain-option-label"><strong>{{ name }}</strong><b>{{ percent(probability) }}</b></div>
            <div class="brain-bar"><span :style="{ width: percent(probability) }"></span></div>
          </li>
        </ul>
        <span class="stream-ad-url">{{ percent(confidence) }} confidence · {{ probabilities.length }} actions</span>
        <dl class="brain-ram">
          <template v-for="[name, value] in ramState" :key="name">
            <dt>{{ label(name) }}</dt><dd :class="{ flash: changedFields.has(name) }">{{ readable(value) }}</dd>
          </template>
        </dl>
      </aside>

      <div class="stream-bottom">
        <ProgressPath :curriculum="snapshot.metadata?.curricula?.[level] || {}" :playing="live.training_state || ''" />
        <section class="site-card">
          <span class="sight-title">Experiment</span>
          <strong class="site-url">Learning for {{ learningFor }}</strong>
          <span class="site-score">{{ fmt(summary.episodes) }} {{ summary.episodes === 1 ? 'attempt' : 'attempts' }} · best reward {{ fmt(summary.best_fitness, 1) }}</span>
          <time class="site-clock">{{ clock.format(now) }}</time>
        </section>
      </div>
    </div>
  </div>
</template>
