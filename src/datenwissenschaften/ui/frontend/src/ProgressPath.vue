<script setup>
import { computed } from 'vue'
import { words } from './naming.js'
import { pathNodes, splitDelta, splitTime } from './progressPath.js'

const props = defineProps({
  curriculum: { type: Object, required: true },
  playing: { type: String, required: true },
  levels: { type: Object, required: true },
  times: { type: Object, required: true },
  running: { type: [Object, null], required: true },
})

const nodes = computed(() => pathNodes(props.curriculum, props.levels))
const winWidth = phase => `${Math.round(Math.min(1, phase.wins / phase.win_target) * 100)}%`
const note = phase => {
  if (phase.mastered) return 'mastered'
  const start = phase.has_checkpoint ? 'checkpoint ready' : 'no checkpoint yet'
  return phase.active ? `practising · ${phase.wins} / ${phase.win_target} wins` : start
}
const levelNote = (name, phase) => (phase.mastered ? 'beaten' : `whole level · ${phase.wins} / ${phase.win_target}`)
const best = name => props.times[name]?.best_seconds ?? null
const last = name => props.times[name]?.last_seconds ?? null
const runningHere = name => props.running?.level === name
</script>

<template>
  <section class="path-panel">
    <span class="path-title">Curriculum · this attempt starts at {{ playing ? words(playing) : 'power-on' }}</span>
    <ol class="path-track">
      <li
        v-for="(node, index) in nodes"
        :key="node.name"
        :class="['path-node', { reached: node.phase.active || node.phase.mastered, current: node.phase.active, mastered: node.phase.mastered, level: node.level }]"
      >
        <span class="path-dot">{{ node.phase.mastered ? '✓' : index + 1 }}</span>
        <strong class="path-label">{{ words(node.name) }}</strong>
        <span class="path-skill"><span :style="{ width: winWidth(node.phase) }"></span></span>
        <small v-if="!node.level" class="path-note">{{ note(node.phase) }}</small>
        <span v-else class="path-split">
          <b class="split-time">{{ runningHere(node.name) ? splitTime(running.seconds) : splitTime(best(node.name)) }}</b>
          <small v-if="runningHere(node.name)" class="path-note">running · best {{ splitTime(best(node.name)) }}</small>
          <small v-else-if="splitDelta(last(node.name), best(node.name))" :class="['split-delta', { ahead: splitDelta(last(node.name), best(node.name)).ahead }]">
            last {{ splitDelta(last(node.name), best(node.name)).text }}
          </small>
          <small v-else class="path-note">{{ levelNote(node.name, node.phase) }}</small>
        </span>
      </li>
    </ol>
  </section>
</template>
