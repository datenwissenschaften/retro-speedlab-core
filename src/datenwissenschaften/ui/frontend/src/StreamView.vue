<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ProgressPath from './ProgressPath.vue'
import { createArrivalTracker, holdsBest, playingLine, recentAttempts, status } from './attempts.js'
import { layaInputs } from './knowledge.js'
import { fmt, gameTitle, percent, words } from './naming.js'
import { createObsControl, STREAM_TIME_ZONE } from './obsControl.js'
import { levelOf } from './progressPath.js'
import { createReplayPlayer } from './replayPlayer.js'
import { elapsed, minutesLeft } from './runtime.js'
import SpotlightPanel from './SpotlightPanel.vue'
import './stream.css'

const STAGE_WIDTH = 1280
const STAGE_HEIGHT = 720
const SNAPSHOT_INTERVAL_MS = 1500
const RELOAD_SETTLE_MS = 15000
const RELOAD_RETRY_MS = 5000
const CLOCK_INTERVAL_MS = 1000
const RECENT_ATTEMPTS = 8
const LAB_LINE_MS = 7000
const LAB_LINES = [
  'Reading walkthroughs and watching speedruns of the game',
  'Measuring the game’s memory, frame by frame',
  'Writing new goals and rewards for Laya',
  'Testing every change before Laya gets it',
  'Laya rests now and trains on the upgraded game next',
]
const clock = new Intl.DateTimeFormat('en-GB', { timeZone: STREAM_TIME_ZONE, dateStyle: 'medium', timeStyle: 'medium' })

const live = ref({})
const screen = ref(null)
const replayEpisode = ref(null)
const latestEpisode = ref(null)
const recentScores = ref([])
const bestRefresh = ref(0)
const waiting = ref(true)
const replayProgress = ref(0)
const snapshot = ref({ metadata: {}, summary: {} })
const connected = ref(false)
const scale = ref(1)
const now = ref(new Date())
let snapshotTimer
let clockTimer
let labLineTimer
const labLine = ref(0)
const arrived = createArrivalTracker()
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
const showFrame = frame => {
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
  await pause(RELOAD_SETTLE_MS)
  while (!(await pageServed())) await pause(RELOAD_RETRY_MS)
  window.location.reload()
}
const reloadIfPending = () => { if (reloadPending) reload() }
const player = createReplayPlayer({
  video: () => screen.value,
  onFrame: showFrame,
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
  onLatest: (episode, summary) => {
    latestEpisode.value = episode
    recentScores.value = summary.recent_scores || []
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
const labRun = computed(() => snapshot.value.metadata?.lab_run?.active ? snapshot.value.metadata.lab_run : null)
let streaming = false
let pausedForLab = false
watch([twitch, labRun], ([enabled, lab]) => {
  if (lab && !pausedForLab) {
    pausedForLab = true
    player.stop()
    live.value = {}
    replayEpisode.value = null
    replayProgress.value = 0
    return
  }
  if (!lab && pausedForLab) {
    reload()
    return
  }
  if (!enabled || lab || streaming) return
  streaming = true
  player.start(); obsControl.start()
})

onMounted(() => {
  fit(); loadSnapshot()
  window.addEventListener('resize', fit)
  snapshotTimer = window.setInterval(loadSnapshot, SNAPSHOT_INTERVAL_MS)
  clockTimer = window.setInterval(() => { now.value = new Date() }, CLOCK_INTERVAL_MS)
  labLineTimer = window.setInterval(() => { labLine.value = (labLine.value + 1) % LAB_LINES.length }, LAB_LINE_MS)
})
onBeforeUnmount(() => {
  player.stop(); obsControl.stop()
  window.removeEventListener('resize', fit)
  window.clearInterval(snapshotTimer); window.clearInterval(clockTimer); window.clearInterval(labLineTimer)
})

const release = computed(() => snapshot.value.server?.release || null)
const run = computed(() => snapshot.value.metadata?.run || {})
const level = computed(() => live.value.level || run.value.savestate || '')
const summary = computed(() => snapshot.value.summary?.by_savestate?.[level.value] || {})
const story = computed(() => snapshot.value.metadata?.stories?.[level.value] || { phases: [], danger: [] })
const levelBest = savestate => snapshot.value.summary?.by_savestate?.[savestate]?.best_fitness ?? null
const recent = computed(() => recentAttempts(latestEpisode.value, recentScores.value, levelBest(latestEpisode.value?.result.level), RECENT_ATTEMPTS))
const replayStatus = computed(() => labRun.value ? 'Upgrading' : status(connected.value, replayEpisode.value))
const replayIsBest = computed(() => replayEpisode.value !== null && holdsBest(replayEpisode.value, levelBest(replayEpisode.value.result.level)))
const learningFor = computed(() => snapshot.value.started_at ? elapsed(snapshot.value.started_at, now.value) : '—')
const inputs = computed(() => layaInputs(snapshot.value.metadata?.knowledge, snapshot.value.metadata?.model?.laya))
const levels = computed(() => snapshot.value.metadata?.environment?.levels || {})
const levelRun = computed(() => {
  const episode = replayEpisode.value
  if (!episode || !(episode.result.curriculum in levels.value)) return null
  return { level: episode.result.curriculum, seconds: replayProgress.value * episode.frame_count / episode.frame_rate }
})
const agentName = computed(() => snapshot.value.metadata?.model?.display_name || '—')
const areasReached = computed(() => story.value.phases.filter(item => item.reached).length)
const probabilities = computed(() => Object.entries(live.value.probabilities || {}))
const confidence = computed(() => Math.max(0, ...probabilities.value.map(([, p]) => p)))
const isSighting = value => Boolean(value) && typeof value === 'object' && 'visible' in value
const ramState = computed(() => Object.entries(live.value.ram || {})
  .filter(([name, value]) => !isSighting(value) && name !== 'snake_visible'))
const RAM_ROWS = 10
const hiddenRamRows = computed(() => Math.max(0, ramState.value.length - RAM_ROWS))

watch(release, current => {
  if (!current) return
  if (loadedRelease === null) {
    loadedRelease = current
    return
  }
  if (current === loadedRelease || reloadPending) return
  reloadPending = true
  if (waiting.value) reload()
})
</script>

<template>
  <div class="stream-view">
    <div class="stream-stage" :style="stageStyle">
      <div class="stream-signal-strip" aria-hidden="true"></div>
      <div class="stream-signal-strip bottom" aria-hidden="true"></div>
      <div class="stream-left-rail">
        <aside class="run-info-panel">
          <div class="run-info-brand">
            <img class="run-info-logo" src="/laya-logo.svg" alt="" />
            <span class="run-info-wordmark">
              <strong>Retro Speedlab</strong>
              <small>Cartridge 01 · live signal</small>
            </span>
          </div>
          <div class="run-info-medal">
            <span class="signal-bars" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></span>
            <span>
              <span class="run-info-medal-head">
                <strong class="run-info-medal-title">Attempt {{ replayEpisode ? `#${replayEpisode.result.attempt}` : '—' }}</strong>
                <span class="run-info-kicker">{{ replayStatus }}</span>
              </span>
              <small v-if="replayEpisode" class="run-info-medal-start">{{ words(replayEpisode.result.level) }}</small>
              <span v-if="replayIsBest" class="replay-badge best">★ Best so far</span>
              <span class="replay-track"><span :style="{ width: percent(replayProgress) }"></span></span>
            </span>
          </div>
          <dl class="run-info-grid">
            <div class="run-info-row"><dt>Game</dt><dd>{{ run.game ? gameTitle(run.game) : 'Waiting' }}</dd></div>
            <div class="run-info-row"><dt>Level</dt><dd>{{ levelOf(live.training_state, levels) }}</dd></div>
            <div class="run-info-row"><dt>Curriculum</dt><dd>{{ playingLine(live.training_state, replayEpisode) }}</dd></div>
            <div class="run-info-row"><dt>Agent</dt><dd>{{ agentName }}</dd></div>
          </dl>
        </aside>

        <SpotlightPanel :danger="story.danger" :failures="story.failures || 0" :level="level" :recent="recent" :refresh="bestRefresh" :inputs="inputs" />
      </div>

      <div class="stream-screen">
        <video ref="screen" class="stream-video crt-picture" aria-label="Replayed gameplay" muted playsinline></video>
        <div class="crt-glass" aria-hidden="true"></div>
        <Transition name="fade">
          <div v-if="snapshot.server && !twitch" class="stream-waiting">
            <span class="stream-waiting-kicker">Stream off</span>
            <strong class="stream-waiting-title">Twitch is disabled</strong>
            <span class="stream-waiting-copy">Set twitch.enabled to true in config.yaml to stream the experiment.</span>
          </div>
          <div v-else-if="labRun" class="stream-waiting lab-run">
            <span class="stream-waiting-kicker">Lab run in progress</span>
            <strong class="stream-waiting-title">The lab is upgrading the game<span class="stream-waiting-dots"><i>.</i><i>.</i><i>.</i></span></strong>
            <Transition name="fade" mode="out-in"><span :key="labLine" class="stream-waiting-copy lab-run-line">{{ LAB_LINES[labLine] }}</span></Transition>
            <span class="stream-waiting-copy">{{ agentName }} pauses training until the upgrade is done · back in about {{ minutesLeft(labRun.until, now) }} min</span>
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

      <aside v-if="labRun" class="stream-ad-panel brain-panel">
        <strong class="stream-ad-title">Lab upgrade</strong>
        <span class="stream-ad-copy">{{ agentName }}'s next moves return when training resumes</span>
        <ol class="brain-options lab-steps">
          <li v-for="(line, index) in LAB_LINES" :key="line" :class="{ chosen: index === labLine }">{{ line }}</li>
        </ol>
        <span class="stream-ad-url">Back in about {{ minutesLeft(labRun.until, now) }} min</span>
      </aside>
      <aside v-else class="stream-ad-panel brain-panel">
        <strong class="stream-ad-title">Next move</strong>
        <span class="stream-ad-copy">{{ probabilities.length ? `Choosing from ${probabilities.length} actions` : 'Waiting for the first finished attempt…' }}</span>
        <ul class="brain-options">
          <li v-for="[name, probability] in probabilities" :key="name" :class="{ chosen: name === live.action }">
            <div class="brain-option-label"><strong>{{ name }}</strong><b>{{ percent(probability) }}</b></div>
            <div class="brain-bar"><span :style="{ width: percent(probability) }"></span></div>
          </li>
        </ul>
        <span class="stream-ad-url">{{ percent(confidence) }} confidence · {{ probabilities.length }} actions</span>
        <div :class="['brain-ram-window', { scrolling: hiddenRamRows > 0 }]" :style="{ '--hidden-rows': hiddenRamRows }">
          <dl class="brain-ram">
            <template v-for="[name, value] in ramState" :key="name">
              <dt>{{ label(name) }}</dt><dd>{{ readable(value) }}</dd>
            </template>
          </dl>
        </div>
      </aside>

      <div class="stream-bottom">
        <ProgressPath
          :curriculum="snapshot.metadata?.curricula?.[level] || {}"
          :playing="live.training_state || ''"
          :levels="levels"
          :times="snapshot.metadata?.level_times || {}"
          :running="levelRun"
          :models="snapshot.metadata?.state_models || {}"
        />
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
