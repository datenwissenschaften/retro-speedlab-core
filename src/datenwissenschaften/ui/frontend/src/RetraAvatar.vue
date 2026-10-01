<script setup>
import { computed } from 'vue'

const props = defineProps({
  expression: { type: String, required: true },
  name: { type: String, required: true },
  tag: { type: String, required: true },
})

const BROWS = { focused: [-3, 0], happy: [2, -2], cheer: [4, -4], shocked: [4, -7], sad: [10, 3] }
const closedEyes = computed(() => props.expression === 'happy' || props.expression === 'cheer')
const irisScale = computed(() => (props.expression === 'shocked' ? 0.78 : 1))
const blush = computed(() => props.expression !== 'focused')
const browTilt = computed(() => BROWS[props.expression][0])
const browLift = computed(() => BROWS[props.expression][1])
</script>

<template>
  <figure :class="['retra', expression]">
    <svg viewBox="0 0 300 300" role="img" :aria-label="name">
      <defs>
        <radialGradient id="retra-backdrop" cx="50%" cy="38%" r="75%">
          <stop offset="0%" stop-color="#3a0d18" />
          <stop offset="60%" stop-color="#14070c" />
          <stop offset="100%" stop-color="#050307" />
        </radialGradient>
        <linearGradient id="retra-hair" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2e2a36" />
          <stop offset="45%" stop-color="#16141b" />
          <stop offset="100%" stop-color="#060508" />
        </linearGradient>
        <linearGradient id="retra-streak" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#ff3b52" />
          <stop offset="55%" stop-color="#c8142c" />
          <stop offset="100%" stop-color="#6e0815" />
        </linearGradient>
        <linearGradient id="retra-skin" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#fde8da" />
          <stop offset="100%" stop-color="#f3cdb8" />
        </linearGradient>
        <linearGradient id="retra-iris" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2a1208" />
          <stop offset="45%" stop-color="#6b3517" />
          <stop offset="100%" stop-color="#c47a3c" />
        </linearGradient>
        <linearGradient id="retra-leather" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#3a3a44" />
          <stop offset="35%" stop-color="#16161c" />
          <stop offset="100%" stop-color="#060609" />
        </linearGradient>
        <linearGradient id="retra-sheen" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#fff" stop-opacity="0" />
          <stop offset="50%" stop-color="#fff" stop-opacity="0.28" />
          <stop offset="100%" stop-color="#fff" stop-opacity="0" />
        </linearGradient>
        <filter id="retra-soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3" /></filter>
        <clipPath id="retra-face"><path d="M102 128 Q102 178 130 204 Q142 214 150 215 Q158 214 170 204 Q198 178 198 128 Q198 76 150 72 Q102 76 102 128 Z" /></clipPath>
        <clipPath id="retra-eye"><path d="M-17 3 Q-15 -12 1 -13 Q15 -12 19 -2 Q17 12 2 14 Q-13 13 -17 3 Z" /></clipPath>
        <path id="retra-bangs" d="M94 160 Q84 64 150 54 Q216 64 206 160 Q203 140 197 128 Q197 144 190 156 Q190 132 182 116 Q180 134 172 146 Q173 124 166 108 Q162 128 154 142 Q156 122 150 106 Q144 124 136 140 Q138 120 134 108 Q128 128 120 144 Q122 126 126 112 Q114 128 108 150 Q106 136 106 124 Q98 140 94 160 Z" />
        <g id="retra-open-eye">
          <path d="M-17 3 Q-15 -12 1 -13 Q15 -12 19 -2 Q17 12 2 14 Q-13 13 -17 3 Z" fill="#fff" />
          <g clip-path="url(#retra-eye)">
            <g :transform="`translate(1 2) scale(${irisScale}) translate(-1 -2)`">
              <ellipse cx="1" cy="2" rx="11.5" ry="14" fill="url(#retra-iris)" />
              <ellipse cx="1" cy="3" rx="5.5" ry="7.5" fill="#160703" />
              <ellipse cx="1" cy="9" rx="7" ry="3.5" fill="#e8a060" opacity="0.45" />
            </g>
            <ellipse cx="1" cy="-12" rx="22" ry="9" fill="#2a1410" opacity="0.35" />
          </g>
          <ellipse cx="6" cy="-4" rx="3.6" ry="4.2" fill="#fff" />
          <circle cx="-4" cy="8" r="1.7" fill="#fff" opacity="0.85" />
          <path d="M-19 4 Q-17 -14 1 -16 Q16 -15 21 -5 L25 -9 L23 -2 Q17 -11 1 -11.5 Q-14 -11 -16 5 Z" fill="#120c0c" />
          <path d="M-9 13.5 Q3 16 14 10" stroke="#7a4434" stroke-width="1.2" stroke-linecap="round" fill="none" />
        </g>
        <g id="retra-closed-eye">
          <path d="M-17 2 Q1 -12 19 0" stroke="#120c0c" stroke-width="3.6" stroke-linecap="round" fill="none" />
          <path d="M18 -1 L24 -5" stroke="#120c0c" stroke-width="2.4" stroke-linecap="round" />
        </g>
      </defs>

      <rect width="300" height="300" fill="url(#retra-backdrop)" />
      <g class="body">
        <path d="M150 36 Q226 40 218 128 Q224 216 250 300 L50 300 Q76 216 82 128 Q74 40 150 36 Z" fill="url(#retra-hair)" />
        <path d="M134 198 L166 198 L168 240 Q150 250 132 240 Z" fill="#f1c7b1" />
        <path d="M134 204 Q150 226 166 204 L167 220 Q150 236 133 220 Z" fill="#d89c86" opacity="0.6" />

        <path d="M126 232 Q150 244 174 232 L180 300 L120 300 Z" fill="#111116" />
        <path d="M126 226 Q150 238 174 226 L176 246 Q150 258 124 246 Z" fill="#1d1d24" stroke="#2c2c36" stroke-width="1" />
        <path d="M150 240 L150 300" stroke="#9a9aa8" stroke-width="2" stroke-dasharray="2 1.5" />
        <rect x="147" y="252" width="6" height="11" rx="2" fill="#c9c9d4" />
        <circle cx="150" cy="266" r="3" fill="none" stroke="#c8142c" stroke-width="1.6" />

        <path d="M28 300 Q30 262 80 248 Q108 240 126 236 L136 300 Z" fill="url(#retra-leather)" />
        <path d="M272 300 Q270 262 220 248 Q192 240 174 236 L164 300 Z" fill="url(#retra-leather)" />
        <path d="M126 236 L104 258 L122 270 L110 280 L136 300 Z" fill="#1f1f27" stroke="#3c3c48" stroke-width="1" />
        <path d="M174 236 L196 258 L178 270 L190 280 L164 300 Z" fill="#1f1f27" stroke="#3c3c48" stroke-width="1" />
        <path d="M48 270 Q70 252 98 248" stroke="url(#retra-sheen)" stroke-width="7" stroke-linecap="round" fill="none" />
        <path d="M252 270 Q230 252 202 248" stroke="url(#retra-sheen)" stroke-width="7" stroke-linecap="round" fill="none" />
        <path d="M60 300 Q66 280 84 268 M240 300 Q234 280 216 268" stroke="#3c3c48" stroke-width="1" stroke-dasharray="3 2" fill="none" />

        <path d="M96 60 Q78 140 80 230 M204 60 Q222 140 220 230 M112 50 Q92 120 90 200 M188 50 Q208 120 210 200" stroke="#2f2b37" stroke-width="1.4" fill="none" opacity="0.8" />
        <path d="M100 118 Q86 196 92 252 Q96 280 84 300 L108 300 Q114 262 110 214 Q108 168 114 130 Z" fill="url(#retra-hair)" />
        <path d="M200 118 Q214 196 208 252 Q204 280 216 300 L192 300 Q186 262 190 214 Q192 168 186 130 Z" fill="url(#retra-hair)" />

        <g class="head">
          <path d="M102 128 Q102 178 130 204 Q142 214 150 215 Q158 214 170 204 Q198 178 198 128 Q198 76 150 72 Q102 76 102 128 Z" fill="url(#retra-skin)" />
          <g clip-path="url(#retra-face)">
            <use href="#retra-bangs" transform="translate(0 7)" fill="#c98a74" opacity="0.55" filter="url(#retra-soft)" />
          </g>
          <path d="M196 150 Q206 148 205 162 Q203 174 194 174 Z" fill="#efc2ab" />

          <g v-if="blush" class="blush">
            <ellipse cx="117" cy="176" rx="12" ry="5" fill="#ff7f86" opacity="0.38" />
            <ellipse cx="183" cy="176" rx="12" ry="5" fill="#ff7f86" opacity="0.38" />
            <path d="M110 178 L114 173 M116 178 L120 173 M122 178 L126 173 M174 178 L178 173 M180 178 L184 173 M186 178 L190 173" stroke="#e8606c" stroke-width="1" stroke-linecap="round" opacity="0.7" />
          </g>

          <g class="brows" :transform="`translate(0 ${browLift})`">
            <path :transform="`rotate(${-browTilt} 126 127)`" d="M110 129 Q124 121 140 126" stroke="#1d1414" stroke-width="2.6" stroke-linecap="round" fill="none" />
            <path :transform="`rotate(${browTilt} 174 127)`" d="M160 126 Q176 121 190 129" stroke="#1d1414" stroke-width="2.6" stroke-linecap="round" fill="none" />
          </g>

          <g class="eyes">
            <use :href="closedEyes ? '#retra-closed-eye' : '#retra-open-eye'" transform="translate(124 152) scale(-1 1)" />
            <use :href="closedEyes ? '#retra-closed-eye' : '#retra-open-eye'" transform="translate(176 152)" />
          </g>
          <g v-if="expression === 'sad'">
            <path d="M102 144 L146 134 L146 126 L102 126 Z M198 144 L154 134 L154 126 L198 126 Z" fill="url(#retra-skin)" />
            <path d="M104 144 L144 134.5 M196 144 L156 134.5" stroke="#120c0c" stroke-width="3" stroke-linecap="round" />
          </g>

          <path d="M151 170 Q148 177 152 179" stroke="#cf9580" stroke-width="1.6" stroke-linecap="round" fill="none" />

          <path v-if="expression === 'focused'" d="M143 192 Q150 194.5 158 191" stroke="#a8505a" stroke-width="2.2" stroke-linecap="round" fill="none" />
          <path v-else-if="expression === 'happy'" d="M139 189 Q150 199 161 189" stroke="#a8505a" stroke-width="2.4" stroke-linecap="round" fill="none" />
          <g v-else-if="expression === 'cheer'">
            <path d="M137 187 Q150 208 163 187 Q150 191 137 187 Z" fill="#7a1f2b" />
            <path d="M143 199 Q150 205 157 199 Q150 197 143 199 Z" fill="#f08390" />
            <path d="M139 188 Q150 191 161 188 L160 190.5 Q150 193 140 190.5 Z" fill="#fff" />
          </g>
          <ellipse v-else-if="expression === 'shocked'" cx="150" cy="194" rx="5" ry="6.5" fill="#7a1f2b" />
          <path v-else d="M142 196 Q150 190 158 196" stroke="#a8505a" stroke-width="2.2" stroke-linecap="round" fill="none" />

          <g class="hair-front">
            <use href="#retra-bangs" fill="url(#retra-hair)" />
            <path d="M106 116 Q96 160 102 212 Q110 184 112 152 Q113 132 118 120 Z" fill="url(#retra-hair)" />
            <path d="M194 116 Q204 160 198 212 Q190 184 188 152 Q187 132 182 120 Z" fill="url(#retra-hair)" />
            <path d="M140 58 Q120 78 114 108 Q108 140 106 178 Q105 198 102 214 Q114 192 116 160 Q118 126 126 102 Q132 82 146 62 Z" fill="url(#retra-streak)" />
            <path d="M134 70 Q120 92 116 124 Q113 150 111 170" stroke="#ff9aa6" stroke-width="1.1" fill="none" opacity="0.55" />
            <path d="M112 98 Q122 88 136 84 Q124 92 116 102 Z M146 80 Q160 78 172 82 Q160 84 148 86 Z M178 84 Q188 88 194 98 Q186 92 176 90 Z" fill="#8d8798" opacity="0.6" />
          </g>
          <g class="earring">
            <path d="M200 174 L200 182" stroke="#c9c9d4" stroke-width="1.2" />
            <circle cx="200" cy="189" r="6.5" fill="#101014" stroke="#d6d6e0" stroke-width="1.4" />
            <path d="M196.5 192 L197.6 186.6 L200 190 L202.4 186.6 L203.5 192" stroke="#d6d6e0" stroke-width="1" fill="none" stroke-linejoin="round" />
          </g>
        </g>
      </g>
    </svg>
    <figcaption><strong>{{ name }}</strong><span>{{ tag }}</span><i class="live">LIVE</i></figcaption>
  </figure>
</template>

<style scoped>
.retra { position: relative; margin: 0; width: 100%; height: 100%; overflow: hidden; border-radius: 12px; border: 1px solid rgba(255, 59, 82, 0.55); background: #050307; box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.6), 0 10px 28px rgba(0, 0, 0, 0.6), 0 0 22px rgba(200, 20, 44, 0.28); }
svg { display: block; width: 100%; height: 100%; }
figcaption { position: absolute; left: 0; right: 0; bottom: 0; display: flex; align-items: baseline; gap: 6px; padding: 14px 10px 7px; background: linear-gradient(transparent, rgba(5, 3, 7, 0.92) 45%); font-family: var(--font-mono); }
figcaption strong { color: #f8fafc; font-size: 0.9rem; letter-spacing: 0.04em; }
figcaption span { color: #ff5a6e; font-size: 0.72rem; }
.live { margin-left: auto; padding: 1px 6px; border-radius: 3px; background: #c8142c; color: #fff; font-size: 0.6rem; font-style: normal; font-weight: 700; letter-spacing: 0.08em; }
.body { animation: breathe 4s ease-in-out infinite; transform-origin: 150px 300px; }
.head { transform-origin: 150px 210px; transition: transform 0.4s ease; }
.eyes { animation: blink 5.5s infinite; transform-origin: 150px 152px; }
.hair-front { animation: sway 6s ease-in-out infinite; transform-origin: 150px 60px; }
.cheer .body { animation: bounce 0.6s ease-in-out infinite; }
.shocked .head { animation: shake 0.35s ease-in-out 3; }
.sad .head { transform: rotate(-3deg) translateY(3px); }
.happy .head { transform: rotate(2deg); }

@keyframes breathe { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-2px); } }
@keyframes blink { 0%, 93%, 100% { transform: scaleY(1); } 95% { transform: scaleY(0.08); } }
@keyframes sway { 0%, 100% { transform: rotate(-0.6deg); } 50% { transform: rotate(0.6deg); } }
@keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-3px); } 75% { transform: translateX(3px); } }
</style>
