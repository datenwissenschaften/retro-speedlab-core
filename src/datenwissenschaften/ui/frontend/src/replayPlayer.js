const POLL_INTERVAL_MS = 1000
const PREROLL_SECONDS = 2
const DECODE_AHEAD_FRAMES = 60

const decode = async image => {
  const bytes = Uint8Array.from(atob(image), character => character.charCodeAt(0))
  return createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }))
}

const fetchJson = async url => {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

export const createReplayPlayer = ({ onFrame, onEpisode, onEpisodeEnd, onWaiting, onSummary, onConnection }) => {
  let generation = null
  let latest = null
  let lastPlayedId = null
  let episode = null
  let frames = []
  let bitmaps = new Map()
  let startedAt = null
  let shown = -1
  let timer
  let animation

  const poll = async () => {
    try {
      const payload = await fetchJson('/api/live/episode')
      onConnection(true)
      onSummary(payload.summary)
      if (payload.generation !== generation) startGeneration(payload.generation)
      latest = payload.episode
      if (!episode && latest && latest.id !== lastPlayedId) begin(latest)
    } catch {
      onConnection(false)
    }
  }

  const release = () => {
    bitmaps.forEach(bitmap => bitmap?.close())
    bitmaps = new Map()
    frames = []
  }

  const startGeneration = next => {
    const interrupted = episode !== null
    generation = next
    lastPlayedId = null
    episode = null
    release()
    if (interrupted) onWaiting()
  }

  const begin = next => {
    release()
    episode = next
    startedAt = null
    shown = -1
    onEpisode(next)
    download(next)
  }

  const finish = () => {
    onEpisodeEnd(episode)
    lastPlayedId = episode.id
    release()
    episode = null
    if (latest && latest.id !== lastPlayedId) begin(latest)
    else onWaiting()
  }

  const download = async target => {
    const source = generation
    try {
      while (episode === target && frames.length < target.frame_count) {
        const payload = await fetchJson(`/api/live/frames?generation=${source}&episode=${target.id}&start=${frames.length}`)
        if (episode !== target) return
        frames.push(...payload.frames)
      }
    } catch {
      if (episode === target) {
        episode = null
        onWaiting()
      }
    }
  }

  const decodeAhead = index => {
    const target = bitmaps
    for (let ahead = index; ahead < Math.min(frames.length, index + DECODE_AHEAD_FRAMES); ahead += 1) {
      if (target.has(ahead)) continue
      target.set(ahead, null)
      decode(frames[ahead].image).then(bitmap => {
        if (bitmaps === target && target.get(ahead) === null) target.set(ahead, bitmap)
        else bitmap.close()
      })
    }
    for (const [past, bitmap] of bitmaps) {
      if (past >= index) continue
      bitmap?.close()
      bitmaps.delete(past)
    }
  }

  const ready = () => frames.length >= Math.min(episode.frame_count, PREROLL_SECONDS * episode.frame_rate)

  const tick = time => {
    animation = requestAnimationFrame(tick)
    if (!episode) return
    if (startedAt === null) {
      decodeAhead(0)
      if (ready() && bitmaps.get(0)) startedAt = time
      return
    }
    const index = Math.floor((time - startedAt) / 1000 * episode.frame_rate)
    if (index >= episode.frame_count) {
      finish()
      return
    }
    if (index >= frames.length) {
      startedAt += time - startedAt - shown / episode.frame_rate * 1000
      return
    }
    decodeAhead(index)
    const bitmap = bitmaps.get(index)
    if (index === shown || !bitmap) return
    const passed = frames.slice(shown + 1, index + 1).map(frame => frame.status)
    shown = index
    onFrame({ bitmap, status: frames[index].status, passed, progress: index / episode.frame_count })
  }

  return {
    start() {
      poll()
      timer = window.setInterval(poll, POLL_INTERVAL_MS)
      animation = requestAnimationFrame(tick)
    },
    stop() {
      window.clearInterval(timer)
      cancelAnimationFrame(animation)
      release()
    },
  }
}
