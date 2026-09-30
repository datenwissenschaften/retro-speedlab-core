<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  level: { type: String, required: true },
  danger: { type: Array, required: true },
})

const SLIDE_MS = 20000
const VIDEO_REFRESH_MS = 60000
const MINUTE_MS = 60000
const videos = ref([])
const slide = ref('best')
let slideTimer
let videoTimer

const loadVideos = async () => {
  try {
    const response = await fetch('/api/rollout-videos', { cache: 'no-store' })
    if (response.ok) videos.value = (await response.json()).videos
  } catch {
    videos.value = []
  }
}

const best = computed(() => videos.value.filter(video => video.savestate === props.level).reduce((top, video) => (!top || video.score > top.score ? video : top), null))
const phaseLabel = name => name.replace(/([a-z])([A-Z])/g, '$1 $2')
const age = recordedAt => {
  const minutes = Math.round((Date.now() - new Date(recordedAt).getTime()) / MINUTE_MS)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  return `${Math.round(minutes / 60)} h ago`
}
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
})
</script>

<template>
  <section class="spotlight-panel">
    <Transition name="fade" mode="out-in">
      <div v-if="slide === 'best'" key="best" class="spotlight-slide">
        <span class="sight-title">Best attempt so far</span>
        <figure v-if="best" class="best-attempt">
          <video :key="best.path" :src="source(best)" autoplay muted loop playsinline></video>
          <figcaption>
            <b>{{ best.score.toFixed(1) }} points</b>
            <span>{{ phaseLabel(best.curriculum) }} · {{ age(best.recorded_at) }}</span>
          </figcaption>
        </figure>
        <span v-else class="game-cover-loading">Collecting the first attempts</span>
      </div>
      <div v-else key="danger" class="spotlight-slide">
        <span class="sight-title">Danger spots</span>
        <ol class="danger-spots">
          <li v-for="(spot, index) in danger" :key="index">
            <img :src="`data:image/jpeg;base64,${spot.image}`" alt="Where attempts fail" />
            <span><b>Failed here {{ spot.count }}×</b>{{ spot.phase }}</span>
          </li>
        </ol>
        <span class="spotlight-note">Where the last 100 failed attempts ended</span>
      </div>
    </Transition>
  </section>
</template>
