<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { dangerNote, dangerTitle } from './attempts.js'
import { words } from './naming.js'

const props = defineProps({
  level: { type: String, required: true },
  danger: { type: Array, required: true },
  failures: { type: Number, required: true },
  refresh: { type: Number, required: true },
})

const SLIDE_MS = 20000
const VIDEO_REFRESH_MS = 60000
const MINUTE_MS = 60000
const FRESH_MS = 8000
const videos = ref([])
const slide = ref('best')
const fresh = ref(false)
let slideTimer
let videoTimer
let freshTimer

const loadVideos = async () => {
  try {
    const response = await fetch('/api/rollout-videos', { cache: 'no-store' })
    if (response.ok) videos.value = (await response.json()).videos
  } catch {
    videos.value = []
  }
}

const best = computed(() => videos.value.filter(video => video.savestate === props.level).reduce((top, video) => (!top || video.score > top.score ? video : top), null))
const age = recordedAt => {
  const minutes = Math.round((Date.now() - new Date(recordedAt).getTime()) / MINUTE_MS)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  return `${Math.round(minutes / 60)} h ago`
}
watch(() => props.refresh, loadVideos)
watch(() => best.value?.path, (current, previous) => {
  if (!current || !previous) return
  slide.value = 'best'
  fresh.value = true
  window.clearTimeout(freshTimer)
  freshTimer = window.setTimeout(() => { fresh.value = false }, FRESH_MS)
})
const source = video => `/api/rollout-video?path=${encodeURIComponent(video.path)}`
const nextSlide = () => {
  slide.value = slide.value === 'best' && props.danger.length ? 'danger' : 'best'
}

onMounted(() => {
  loadVideos()
  videoTimer = window.setInterval(loadVideos, VIDEO_REFRESH_MS)
  slideTimer = window.setInterval(nextSlide, SLIDE_MS)
})
onBeforeUnmount(() => {
  window.clearInterval(videoTimer)
  window.clearInterval(slideTimer)
  window.clearTimeout(freshTimer)
})
</script>

<template>
  <section :class="['spotlight-panel', { fresh }]">
    <Transition name="fade" mode="out-in">
      <div v-if="slide === 'best'" key="best" class="spotlight-slide">
        <span class="sight-title">{{ fresh ? '★ New best attempt' : 'Best attempt so far' }}</span>
        <figure v-if="best" class="best-attempt">
          <video :key="best.path" :src="source(best)" autoplay muted loop playsinline></video>
          <figcaption>
            <b>Reward {{ best.score.toFixed(1) }}</b>
            <span>{{ words(best.curriculum) }} · {{ age(best.recorded_at) }}</span>
          </figcaption>
        </figure>
        <span v-else class="game-cover-loading">Collecting the first attempts</span>
      </div>
      <div v-else key="danger" class="spotlight-slide">
        <span class="sight-title">Danger spots</span>
        <ol class="danger-spots">
          <li v-for="(spot, index) in danger" :key="index">
            <img :src="`data:image/jpeg;base64,${spot.image}`" alt="Last frame of a failed attempt" />
            <span><b>{{ dangerTitle(spot) }}</b>{{ spot.phase }}</span>
          </li>
        </ol>
        <span class="spotlight-note">{{ dangerNote(failures, danger.some(spot => spot.located)) }}</span>
      </div>
    </Transition>
  </section>
</template>
