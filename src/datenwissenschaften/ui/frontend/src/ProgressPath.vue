<script setup>
const props = defineProps({
  phases: { type: Array, required: true },
  current: { type: String, required: true },
  curriculum: { type: Object, required: true },
})

const progress = phase => props.curriculum[phase.name] || { wins: 0, win_target: 1, mastered: false }
const mastered = phase => progress(phase).mastered
const beatenWidth = phase => `${Math.round((progress(phase).wins / progress(phase).win_target) * 100)}%`
const note = phase => {
  if (!phase.reached) return 'not reached yet'
  if (mastered(phase)) return 'mastered'
  return `beaten ${progress(phase).wins} of ${progress(phase).win_target}`
}
</script>

<template>
  <section class="path-panel">
    <span class="path-title">The journey</span>
    <ol class="path-track">
      <li
        v-for="(phase, index) in phases"
        :key="phase.name"
        :class="['path-node', { reached: phase.reached, current: phase.name === current, mastered: mastered(phase) }]"
      >
        <span class="path-dot">{{ mastered(phase) ? '✓' : phase.reached ? index + 1 : '?' }}</span>
        <strong class="path-label">{{ phase.reached ? phase.label : 'Unknown' }}</strong>
        <span class="path-skill"><span :style="{ width: beatenWidth(phase) }"></span></span>
        <small class="path-note">{{ note(phase) }}</small>
      </li>
    </ol>
  </section>
</template>
