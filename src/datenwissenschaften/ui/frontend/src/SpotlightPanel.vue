<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  phase: { type: String, required: true },
  map: { type: Object, required: true },
})

const SLIDE_MS = 20000
const VIDEO_REFRESH_MS = 60000
const videos = ref([])
const slide = ref('then-now')
const canvas = ref(null)
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

const phaseVideos = computed(() => videos.value
  .filter(video => video.curriculum === props.phase)
  .sort((first, second) => String(first.recorded_at).localeCompare(String(second.recorded_at))))
const then = computed(() => phaseVideos.value.length > 1 ? phaseVideos.value[0] : null)
const now = computed(() => phaseVideos.value.at(-1) || null)
const source = video => `/api/rollout-video?path=${encodeURIComponent(video.path)}`
const hasMap = computed(() => props.map.visits.length > 0)
const endCount = computed(() => props.map.ends.reduce((total, [, , count]) => total + count, 0))

const drawMap = () => {
  const element = canvas.value
  if (!element || !hasMap.value) return
  const cells = props.map.visits
  const xs = cells.map(([x]) => x)
  const ys = cells.map(([, y]) => y)
  const [minX, minY] = [Math.min(...xs), Math.min(...ys)]
  const columns = Math.max(...xs) - minX + 1
  const rows = Math.max(...ys) - minY + 1
  const size = Math.max(1, Math.floor(Math.min(element.width / columns, element.height / rows)))
  const [left, top] = [(element.width - columns * size) / 2, (element.height - rows * size) / 2]
  const context = element.getContext('2d')
  context.clearRect(0, 0, element.width, element.height)
  context.fillStyle = 'rgba(41, 240, 255, 0.28)'
  for (const [x, y] of cells) context.fillRect(left + (x - minX) * size, top + (y - minY) * size, size, size)
  const hottest = Math.max(1, ...props.map.ends.map(([, , count]) => count))
  for (const [x, y, count] of props.map.ends) {
    context.fillStyle = `rgba(255, 79, 100, ${0.35 + 0.65 * Math.log1p(count) / Math.log1p(hottest)})`
    context.fillRect(left + (x - minX) * size - 1, top + (y - minY) * size - 1, size + 2, size + 2)
  }
}

const nextSlide = () => {
  slide.value = slide.value === 'then-now' && hasMap.value ? 'map' : 'then-now'
}

watch([slide, () => props.map], () => requestAnimationFrame(drawMap), { deep: true })

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
      <div v-if="slide === 'then-now'" key="then-now" class="spotlight-slide">
        <span class="sight-title">Then vs now</span>
        <div v-if="now" class="then-now">
          <figure v-if="then">
            <video :src="source(then)" autoplay muted loop playsinline></video>
            <figcaption>Then · {{ then.score.toFixed(1) }} points</figcaption>
          </figure>
          <figure>
            <video :src="source(now)" autoplay muted loop playsinline></video>
            <figcaption>{{ then ? 'Now' : 'Best so far' }} · {{ now.score.toFixed(1) }} points</figcaption>
          </figure>
        </div>
        <span v-else class="game-cover-loading">Collecting the first attempts</span>
      </div>
      <div v-else key="map" class="spotlight-slide">
        <span class="sight-title">Where attempts end</span>
        <canvas ref="canvas" class="failure-map" width="280" height="190"></canvas>
        <span class="spotlight-note">Red marks {{ endCount }} failed attempts on the explored level</span>
      </div>
    </Transition>
  </section>
</template>
