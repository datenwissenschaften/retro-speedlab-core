<script setup>
import { computed } from 'vue'
import { words } from './naming.js'
import { PATH_WINDOW, pathNodes, pathWindow, splitDelta, splitTime } from './progressPath.js'

const props = defineProps({
  curriculum: { type: Object, required: true },
  playing: { type: String, required: true },
  levels: { type: Object, required: true },
  times: { type: Object, required: true },
  running: { type: [Object, null], required: true },
  models: { type: Object, required: true },
})

const nodes = computed(() => pathWindow(pathNodes(props.curriculum, props.levels), PATH_WINDOW))
const winWidth = phase => `${Math.round(Math.min(1, phase.wins / phase.win_target) * 100)}%`
const stalled = name => props.models[name]?.stalled === true
const note = (name, phase) => {
  if (phase.mastered) return 'mastered'
  if (stalled(name)) return 'stalled'
  if (phase.active) return `${phase.wins} / ${phase.win_target} wins`
  return phase.has_checkpoint ? 'ready' : 'not reached'
}
const levelNote = (name, phase) => (phase.mastered ? 'beaten' : `level · ${phase.wins} / ${phase.win_target}`)
const best = name => props.times[name]?.best_seconds ?? null
const last = name => props.times[name]?.last_seconds ?? null
const runningHere = name => props.running?.level === name
</script>

<template>
  <section class="path-panel">
    <span class="path-title">Curriculum · this attempt starts at {{ playing ? words(playing) : 'power-on' }}</span>
    <ol class="path-track">
      <template v-for="node in nodes" :key="node.key || node.name">
      <li v-if="node.gap" class="path-gap" aria-label="More curriculum states">…</li>
      <li
        v-else
        :class="['path-node', { reached: node.phase.active || node.phase.mastered, current: node.phase.active, mastered: node.phase.mastered, level: node.level, stalled: stalled(node.name) }]"
      >
        <span class="path-dot">{{ node.phase.mastered ? '✓' : node.number }}</span>
        <strong class="path-label">{{ words(node.name) }}</strong>
        <span class="path-skill"><span :style="{ width: winWidth(node.phase) }"></span></span>
        <small v-if="!node.level" class="path-note">{{ note(node.name, node.phase) }}</small>
        <span v-else class="path-split">
          <b class="split-time">{{ runningHere(node.name) ? splitTime(running.seconds) : splitTime(best(node.name)) }}</b>
          <small v-if="runningHere(node.name)" class="path-note">running · best {{ splitTime(best(node.name)) }}</small>
          <small v-else-if="splitDelta(last(node.name), best(node.name))" :class="['split-delta', { ahead: splitDelta(last(node.name), best(node.name)).ahead }]">
            last {{ splitDelta(last(node.name), best(node.name)).text }}
          </small>
          <small v-else class="path-note">{{ levelNote(node.name, node.phase) }}</small>
        </span>
      </li>
      </template>
    </ol>
  </section>
</template>
