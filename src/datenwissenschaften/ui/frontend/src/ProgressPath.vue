<script setup>
import { words } from './naming.js'

const props = defineProps({
  savestates: { type: Array, required: true },
  curricula: { type: Object, required: true },
  current: { type: String, required: true },
})

const phases = savestate => Object.values(props.curricula[savestate] || {})
const learning = savestate => phases(savestate).find(phase => !phase.mastered)
const mastered = savestate => phases(savestate).length > 0 && !learning(savestate)
const clearedWidth = savestate => {
  const all = phases(savestate)
  if (!all.length) return '0%'
  const open = learning(savestate)
  const done = all.filter(phase => phase.mastered).length + (open ? open.wins / open.win_target : 0)
  return `${Math.round((done / all.length) * 100)}%`
}
const note = savestate => {
  if (mastered(savestate)) return 'mastered'
  const status = savestate === props.current ? 'learning' : 'not active'
  const open = learning(savestate)
  if (!open) return status
  const all = phases(savestate)
  const part = all.length > 1 ? ` · part ${all.indexOf(open) + 1}/${all.length}` : ''
  return `${status} · ${open.wins} / ${open.win_target} clears${part}`
}
</script>

<template>
  <section class="path-panel">
    <span class="path-title">Progress</span>
    <ol class="path-track">
      <li
        v-for="(savestate, index) in savestates"
        :key="savestate"
        :class="['path-node', { reached: savestate === current || mastered(savestate), current: savestate === current, mastered: mastered(savestate) }]"
      >
        <span class="path-dot">{{ mastered(savestate) ? '✓' : index + 1 }}</span>
        <strong class="path-label">{{ words(savestate) }}</strong>
        <span class="path-skill"><span :style="{ width: clearedWidth(savestate) }"></span></span>
        <small class="path-note">{{ note(savestate) }}</small>
      </li>
    </ol>
  </section>
</template>
