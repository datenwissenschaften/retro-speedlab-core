const HUNDREDTHS_PER_SECOND = 100
const HUNDREDTHS_PER_MINUTE = 6000
const ROUNDING_SLACK = 1e-6

const allMastered = (curriculum, states) => states.every(state => curriculum[state]?.mastered)

export const pathNodes = (curriculum, levels) => {
  const beaten = Object.entries(levels).filter(([, states]) => allMastered(curriculum, states))
  const hidden = new Set(beaten.flatMap(([, states]) => states))
  const shown = new Set(beaten.map(([level]) => level))
  return Object.entries(curriculum)
    .filter(([name]) => (name in levels ? shown.has(name) : !hidden.has(name)))
    .map(([name, phase]) => ({ name, phase, level: name in levels }))
}

export const splitTime = seconds => {
  if (seconds == null) return '—'
  const hundredths = Math.floor(seconds * HUNDREDTHS_PER_SECOND + ROUNDING_SLACK)
  const minutes = Math.floor(hundredths / HUNDREDTHS_PER_MINUTE)
  const rest = ((hundredths % HUNDREDTHS_PER_MINUTE) / HUNDREDTHS_PER_SECOND).toFixed(2).padStart(5, '0')
  return `${minutes}:${rest}`
}

export const splitDelta = (seconds, best) => {
  if (seconds == null || best == null) return null
  const delta = seconds - best
  return { text: `${delta < 0 ? '−' : '+'}${Math.abs(delta).toFixed(2)}`, ahead: delta <= 0 }
}
