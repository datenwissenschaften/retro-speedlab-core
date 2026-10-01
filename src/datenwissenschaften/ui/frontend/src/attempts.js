import { words } from './naming.js'

const MIN_BAR_WIDTH = 0.08

export const createArrivalTracker = () => {
  let last = null
  return (generation, episode, replay) => {
    const key = `${generation}:${episode.id}`
    if (replay || key === last) return false
    last = key
    return true
  }
}

const alignedScores = (episode, recentScores) =>
  recentScores?.length && recentScores.at(-1) === episode.result.score ? recentScores : null

export const arrivalOutcome = (episode, recentScores) => {
  const { result } = episode
  if (result.won) return { kind: 'won' }
  if (result.new_best) return { kind: 'best' }
  const scores = alignedScores(episode, recentScores)
  if (scores && scores.length > 1 && result.score > scores.at(-2)) return { kind: 'improved', previous: scores.at(-2) }
  return { kind: 'attempt' }
}

export const recentAttempts = (episode, recentScores, levelBest, count) => {
  const scores = episode ? alignedScores(episode, recentScores) : null
  if (!scores) return []
  const top = Math.max(...scores, levelBest ?? -Infinity)
  const bottom = Math.min(...scores)
  const span = top - bottom
  const latest = episode.result.attempt
  return scores.slice(-count).map((score, index, shown) => ({
    attempt: latest - (shown.length - 1 - index),
    score,
    width: span > 0 ? MIN_BAR_WIDTH + (1 - MIN_BAR_WIDTH) * (score - bottom) / span : 1,
    best: levelBest != null && score >= levelBest,
  })).reverse()
}

export const newlyMastered = (before, after) => {
  if (!before || !after) return []
  return Object.entries(after).flatMap(([savestate, phases]) => Object.entries(phases)
    .filter(([phase, progress]) => progress.mastered && before[savestate]?.[phase]?.mastered === false)
    .map(([phase, progress]) => ({ savestate, phase, phases: Object.keys(phases).length, wins: progress.wins, winTarget: progress.win_target })))
}

const reward = score => `reward ${score.toFixed(1)}`

export const holdsBest = (episode, levelBest) => levelBest != null && episode.result.score >= levelBest

export const arrivalBanner = (episode, recentScores) => {
  const outcome = arrivalOutcome(episode, recentScores)
  const { attempt, score } = episode.result
  if (outcome.kind === 'won') return { kind: 'won', title: '★ Attempt won', detail: `#${attempt} · ${reward(score)}` }
  if (outcome.kind === 'best') return { kind: 'best', title: '★ New best attempt', detail: `#${attempt} · ${reward(score)}` }
  const detail = outcome.kind === 'improved' ? `${reward(score)} · up from ${outcome.previous.toFixed(1)}` : reward(score)
  return { kind: 'arrival', title: `New attempt · #${attempt}`, detail }
}

export const resultBanner = (episode, levelBest) => {
  const { attempt, score, won } = episode.result
  if (won) return { kind: 'won', title: '★ Attempt won', detail: `#${attempt} · ${reward(score)}` }
  if (holdsBest(episode, levelBest)) return { kind: 'best', title: `Result · #${attempt}`, detail: `${reward(score)} · ★ best so far` }
  return { kind: 'result', title: `Result · #${attempt}`, detail: reward(score) }
}

export const status = (connected, banner, replayed) => {
  if (!connected) return 'Offline'
  if (banner?.arrival) return 'New'
  return replayed ? 'Replay' : 'Waiting'
}

export const inProgressLine = (replayed, inProgress) => {
  if (!inProgress) return null
  const { attempt, level } = inProgress
  if (replayed && replayed.result.level === level && replayed.result.attempt + 1 === attempt) return 'Next attempt in progress'
  const where = replayed && replayed.result.level !== level ? `${words(level)} · ` : ''
  return `${where}Attempt #${attempt} in progress`
}

export const dangerTitle = spot => spot.located ? `Failed here ${spot.count}×` : `Failed ${spot.count}×`

export const dangerNote = (failures, located) => {
  const attempts = `last ${failures} failed ${failures === 1 ? 'attempt' : 'attempts'}`
  return located ? `Where the ${attempts} ended` : `From the ${attempts}`
}
