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

export const holdsBest = (episode, levelBest) => levelBest != null && episode.result.score >= levelBest

export const status = (connected, replayed) => {
  if (!connected) return 'Offline'
  return replayed ? 'Replay' : 'Waiting'
}

export const dangerTitle = spot => spot.located ? `Failed here ${spot.count}×` : `Failed ${spot.count}×`

export const dangerNote = (failures, located) => {
  const attempts = `last ${failures} failed ${failures === 1 ? 'attempt' : 'attempts'}`
  return located ? `Where the ${attempts} ended` : `From the ${attempts}`
}

export const playingLine = (state, episode) => {
  if (!state) return '—'
  return episode?.result.full_run ? `${words(state)} · full run` : words(state)
}
