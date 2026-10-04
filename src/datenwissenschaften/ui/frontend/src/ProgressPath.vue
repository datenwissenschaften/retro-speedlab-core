<script setup>
import { words } from './naming.js'

const props = defineProps({
  curriculum: { type: Object, required: true },
  playing: { type: String, required: true },
})

const winWidth = phase => `${Math.round(Math.min(1, phase.wins / phase.win_target) * 100)}%`
const note = phase => {
  if (phase.mastered) return 'mastered'
  const start = phase.has_checkpoint ? 'checkpoint ready' : 'no checkpoint yet'
  return phase.active ? `practising · ${phase.wins} / ${phase.win_target} wins` : start
}
</script>

<template>
  <section class="path-panel">
    <span class="path-title">Curriculum · this attempt starts at {{ playing ? words(playing) : 'power-on' }}</span>
    <ol class="path-track">
      <li
        v-for="(phase, name, index) in curriculum"
        :key="name"
        :class="['path-node', { reached: phase.active || phase.mastered, current: phase.active, mastered: phase.mastered }]"
      >
        <span class="path-dot">{{ phase.mastered ? '✓' : index + 1 }}</span>
        <strong class="path-label">{{ words(name) }}</strong>
        <span class="path-skill"><span :style="{ width: winWidth(phase) }"></span></span>
        <small class="path-note">{{ note(phase) }}</small>
      </li>
    </ol>
  </section>
</template>
