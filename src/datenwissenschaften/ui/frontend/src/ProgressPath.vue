<script setup>
defineProps({
  phases: { type: Array, required: true },
  current: { type: String, required: true },
})

const skillWidth = phase => `${Math.round((phase.skill ?? 0) * 100)}%`
const skillLabel = phase => phase.skill == null ? 'no tries yet' : `skill ${Math.round(phase.skill * 100)}%`
</script>

<template>
  <section class="path-panel">
    <span class="path-title">The journey</span>
    <ol class="path-track">
      <li
        v-for="(phase, index) in phases"
        :key="phase.name"
        :class="['path-node', { reached: phase.reached, current: phase.name === current }]"
      >
        <span class="path-dot">{{ phase.reached ? index + 1 : '?' }}</span>
        <strong class="path-label">{{ phase.reached ? phase.label : 'Unknown' }}</strong>
        <span class="path-skill"><span :style="{ width: skillWidth(phase) }"></span></span>
        <small class="path-note">{{ phase.reached ? skillLabel(phase) : 'not reached yet' }}</small>
      </li>
    </ol>
  </section>
</template>
