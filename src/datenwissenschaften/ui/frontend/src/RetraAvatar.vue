<script setup>
import { computed } from 'vue'

const props = defineProps({
  expression: { type: String, required: true },
})

const closedEyes = computed(() => props.expression === 'happy' || props.expression === 'cheer')
const browTilt = computed(() => ({ sad: -9, shocked: 6, focused: 3, happy: 0, cheer: -2 })[props.expression])
const pupilScale = computed(() => (props.expression === 'shocked' ? 0.6 : 1))
const blush = computed(() => props.expression === 'happy' || props.expression === 'cheer')
</script>

<template>
  <svg :class="['retra', expression]" viewBox="0 0 200 220" role="img" aria-label="Retra">
    <defs>
      <radialGradient id="retra-iris" cx="50%" cy="40%" r="60%">
        <stop offset="0%" stop-color="#9b5a2e" />
        <stop offset="70%" stop-color="#5a2f16" />
        <stop offset="100%" stop-color="#2b150a" />
      </radialGradient>
      <linearGradient id="retra-leather" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2a2a31" />
        <stop offset="100%" stop-color="#0c0c10" />
      </linearGradient>
      <linearGradient id="retra-hair" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#26232b" />
        <stop offset="100%" stop-color="#0a090c" />
      </linearGradient>
    </defs>

    <g class="body">
      <path class="hair-back" d="M42 96 Q34 40 100 30 Q166 40 158 96 L170 200 Q100 214 30 200 Z" fill="url(#retra-hair)" />
      <path d="M90 136 L110 136 L113 168 L87 168 Z" fill="#ebbfa5" />
      <path d="M14 220 Q20 172 72 162 L100 176 L128 162 Q180 172 186 220 Z" fill="url(#retra-leather)" />
      <path d="M72 162 L88 200 L100 176 Z M128 162 L112 200 L100 176 Z" fill="#08080b" stroke="#3a3a44" stroke-width="1" />
      <path d="M100 176 L100 220" stroke="#9a9aa5" stroke-width="1.6" stroke-dasharray="2 2" />
      <path d="M40 186 Q52 178 70 176 M160 186 Q148 178 130 176" stroke="#c0182b" stroke-width="2" fill="none" />

      <path d="M56 92 Q56 146 100 154 Q144 146 144 92 Q144 52 100 50 Q56 52 56 92 Z" fill="#f7d9c6" />
      <ellipse cx="55" cy="104" rx="6" ry="10" fill="#efc7b0" />
      <ellipse cx="145" cy="104" rx="6" ry="10" fill="#efc7b0" />
      <g class="earring">
        <circle cx="146" cy="120" r="4.5" fill="#15151a" stroke="#c9c9d2" stroke-width="1" />
        <path d="M143.6 121.8 L144.6 118.4 L146 120.6 L147.4 118.4 L148.4 121.8" stroke="#c9c9d2" stroke-width="0.8" fill="none" />
      </g>

      <g v-if="blush" class="blush">
        <ellipse cx="76" cy="126" rx="9" ry="4" fill="#f28b8b" opacity="0.55" />
        <ellipse cx="124" cy="126" rx="9" ry="4" fill="#f28b8b" opacity="0.55" />
      </g>

      <g class="brows">
        <path :transform="`rotate(${browTilt} 80 84)`" d="M68 85 Q80 79 92 84" stroke="#1b1515" stroke-width="3" stroke-linecap="round" fill="none" />
        <path :transform="`rotate(${-browTilt} 120 84)`" d="M108 84 Q120 79 132 85" stroke="#1b1515" stroke-width="3" stroke-linecap="round" fill="none" />
      </g>

      <g v-if="closedEyes" class="eyes-closed">
        <path d="M68 106 Q80 96 92 106" stroke="#1b1010" stroke-width="3.5" stroke-linecap="round" fill="none" />
        <path d="M108 106 Q120 96 132 106" stroke="#1b1010" stroke-width="3.5" stroke-linecap="round" fill="none" />
      </g>
      <g v-else class="eyes">
        <g v-for="x in [80, 120]" :key="x">
          <ellipse :cx="x" cy="106" rx="12.5" ry="15" fill="#fff" />
          <g :transform="`translate(${x} 108) scale(${pupilScale}) translate(${-x} -108)`">
            <ellipse :cx="x" cy="108" rx="9" ry="12" fill="url(#retra-iris)" />
            <ellipse :cx="x" cy="109" rx="4.2" ry="6" fill="#1d0c05" />
          </g>
          <circle :cx="x + 4" cy="101" r="3" fill="#fff" />
          <circle :cx="x - 4" cy="113" r="1.4" fill="#fff" opacity="0.8" />
          <path :d="`M${x - 14} 98 Q${x} 86 ${x + 14} 98`" stroke="#140c0c" stroke-width="4" stroke-linecap="round" fill="none" />
          <path v-if="expression === 'sad'" :d="`M${x - 13} 99 Q${x} 95 ${x + 13} 99 L${x + 13} 101 Q${x} 98 ${x - 13} 101 Z`" fill="#f7d9c6" />
        </g>
      </g>

      <path v-if="expression === 'focused'" d="M92 136 Q100 138 108 135" stroke="#9c3b45" stroke-width="2.5" stroke-linecap="round" fill="none" />
      <path v-else-if="expression === 'happy'" d="M88 132 Q100 143 112 132" stroke="#9c3b45" stroke-width="2.8" stroke-linecap="round" fill="none" />
      <g v-else-if="expression === 'cheer'">
        <path d="M86 130 Q100 150 114 130 Z" fill="#8f2333" />
        <path d="M92 140 Q100 147 108 140 Q100 143 92 140 Z" fill="#e8737f" />
      </g>
      <ellipse v-else-if="expression === 'shocked'" cx="100" cy="137" rx="5" ry="6.5" fill="#8f2333" />
      <path v-else d="M90 139 Q100 131 110 139" stroke="#9c3b45" stroke-width="2.6" stroke-linecap="round" fill="none" />

      <g class="hair-front">
        <path d="M54 92 Q52 48 100 44 Q148 48 146 92 Q138 70 128 66 L124 86 L114 64 L104 84 L96 62 L86 84 L78 64 L70 86 Q62 74 54 92 Z" fill="url(#retra-hair)" />
        <path d="M54 92 Q46 130 52 172 Q60 136 62 100 Z M146 92 Q154 130 148 172 Q140 136 138 100 Z" fill="url(#retra-hair)" />
        <path d="M84 46 Q70 66 66 92 Q64 104 60 112 Q72 100 74 86 L86 84 Q84 64 92 50 Z" fill="#c0182b" />
        <path d="M82 54 Q74 70 70 92" stroke="#ff4a5f" stroke-width="1" fill="none" opacity="0.7" />
      </g>
    </g>
  </svg>
</template>

<style scoped>
.retra { width: 100%; height: 100%; overflow: visible; filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.55)); }
.body { animation: breathe 4s ease-in-out infinite; transform-origin: 100px 220px; }
.eyes { animation: blink 5.5s infinite; transform-box: fill-box; transform-origin: center; }
.hair-front { animation: sway 6s ease-in-out infinite; transform-box: fill-box; transform-origin: 50% 0; }
.cheer .body { animation: bounce 0.6s ease-in-out infinite; }
.shocked .body { animation: shake 0.35s ease-in-out 3; }
.sad .body { transform: translateY(3px); }

@keyframes breathe { 0%, 100% { transform: translateY(0) scale(1); } 50% { transform: translateY(-2px) scale(1.01); } }
@keyframes blink { 0%, 93%, 100% { transform: scaleY(1); } 95% { transform: scaleY(0.1); } }
@keyframes sway { 0%, 100% { transform: rotate(-0.8deg); } 50% { transform: rotate(0.8deg); } }
@keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
@keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-3px); } 75% { transform: translateX(3px); } }
</style>
