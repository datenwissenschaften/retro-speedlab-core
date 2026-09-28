<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  phase: { type: String, required: true },
  danger: { type: Array, required: true },
})

const SLIDE_MS = 20000
const VIDEO_REFRESH_MS = 60000
const CLIP_SECONDS = 10
const videos = ref([])
const slide = ref('then-now')
const clip = ref('then')
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
const shown = computed(() => clip.value === 'then' && then.value ? then.value : now.value)
const caption = computed(() => {
  if (!then.value) return `Best so far · ${now.value.score.toFixed(1)} points`
  return clip.value === 'then' ? `Then · ${then.value.score.toFixed(1)} points` : `Now · ${now.value.score.toFixed(1)} points`
})
const source = video => `/api/rollout-video?path=${encodeURIComponent(video.path)}`

const switchClip = () => { clip.value = clip.value === 'then' ? 'now' : 'then' }
const limitClip = event => { if (event.target.currentTime >= CLIP_SECONDS) switchClip() }
const nextSlide = () => {
  slide.value = slide.value === 'then-now' && props.danger.length ? 'danger' : 'then-now'
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
      <div v-if="slide === 'then-now'" key="then-now" class="spotlight-slide">
        <span class="sight-title">Then vs now</span>
        <figure v-if="now" class="then-now">
          <video
            :key="shown.path"
            :src="source(shown)"
            autoplay
            muted
            playsinline
            @timeupdate="limitClip"
            @ended="switchClip"
          ></video>
          <figcaption :class="{ then: clip === 'then' && then }">{{ caption }}</figcaption>
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
