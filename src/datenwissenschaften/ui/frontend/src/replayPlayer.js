const POLL_INTERVAL_MS = 1000

const fetchJson = async url => {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

const fetchBlob = async url => {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.blob()
}

export const createReplayPlayer = ({ video, onFrame, onEpisode, onEpisodeEnd, onWaiting, onLatest, onConnection }) => {
  let generation = null
  let latest = null
  let replays = []
  let shownLiveId = null
  let episode = null
  let statuses = []
  let source = null
  let replaying = false
  let timer

  const poll = async () => {
    try {
      const payload = await fetchJson('/api/live/episode')
      onConnection(true)
      if (payload.generation !== generation) startGeneration(payload.generation)
      onLatest(payload.episode, payload.summary, payload.in_progress)
      latest = payload.episode
      replays = payload.replays
      if (!episode && latest && latest.id !== shownLiveId) begin(latest, false)
    } catch {
      onConnection(false)
    }
  }

  const release = () => {
    const element = video()
    element.pause()
    element.removeAttribute('src')
    if (source) URL.revokeObjectURL(source)
    source = null
    statuses = []
  }

  const startGeneration = next => {
    const interrupted = episode !== null
    generation = next
    shownLiveId = null
    episode = null
    release()
    if (interrupted) onWaiting()
  }

  const begin = async (next, replay) => {
    release()
    episode = next
    replaying = replay
    try {
      const loaded = []
      while (loaded.length < next.frame_count) {
        const payload = await fetchJson(`/api/live/statuses?generation=${generation}&episode=${next.id}&start=${loaded.length}`)
        loaded.push(...payload.statuses)
      }
      const blob = await fetchBlob(`/api/live/video?generation=${generation}&episode=${next.id}`)
      if (episode !== next) return
      statuses = loaded
      source = URL.createObjectURL(blob)
      onEpisode(next, replay, generation)
      play()
    } catch {
      if (episode !== next) return
      episode = null
      onWaiting()
    }
  }

  const play = () => {
    const element = video()
    element.src = source
    element.currentTime = 0
    element.requestVideoFrameCallback(showFrame)
    element.play()
  }

  const showFrame = (now, metadata) => {
    if (!episode || !statuses.length) return
    const index = Math.min(statuses.length - 1, Math.round(metadata.mediaTime * episode.frame_rate))
    onFrame({ status: statuses[index], progress: index / statuses.length })
    video().requestVideoFrameCallback(showFrame)
  }

  const finish = () => {
    if (!episode) return
    onEpisodeEnd(episode)
    if (!replaying) shownLiveId = episode.id
    if (latest && latest.id !== shownLiveId) begin(latest, false)
    else if (replays.length) begin(pickReplay(), true)
    else repeat()
  }

  const pickReplay = () => {
    const next = replays.findIndex(replay => replay.id === episode.id) + 1
    return replays[next % replays.length]
  }

  const repeat = () => {
    replaying = true
    onEpisode(episode, true, generation)
    play()
  }

  return {
    start() {
      video().addEventListener('ended', finish)
      poll()
      timer = window.setInterval(poll, POLL_INTERVAL_MS)
    },
    stop() {
      window.clearInterval(timer)
      video().removeEventListener('ended', finish)
      release()
    },
  }
}
