import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createReplayPlayer } from './replayPlayer.js'

const STATUSES = [
  { attempt: 5, action: 'left', probabilities: { left: 0.7, right: 0.3 } },
  { attempt: 5, action: 'right', probabilities: { left: 0.2, right: 0.8 } },
]
const EPISODE = { id: 4, frame_rate: 1, frame_count: STATUSES.length, result: { attempt: 5, level: 'Level1' } }
const LATEST = { generation: 'g', episode: EPISODE, replays: [], summary: {} }

const settle = async () => {
  for (let round = 0; round < 10; round += 1) await new Promise(resolve => { setTimeout(resolve, 0) })
}

const fakeVideo = () => {
  const listeners = {}
  return {
    frameCallbacks: [],
    played: 0,
    src: null,
    currentTime: 0,
    addEventListener: (name, listener) => { listeners[name] = listener },
    removeEventListener: name => { delete listeners[name] },
    removeAttribute: () => {},
    pause: () => {},
    play() { this.played += 1 },
    requestVideoFrameCallback(callback) { this.frameCallbacks.push(callback) },
    frame(mediaTime) { this.frameCallbacks.shift()(0, { mediaTime }) },
    end: () => listeners.ended(),
  }
}

const installBrowser = responses => {
  const requested = []
  globalThis.window = { setInterval: () => 0, clearInterval: () => {} }
  globalThis.URL = { createObjectURL: blob => `blob:${blob}`, revokeObjectURL: () => {} }
  globalThis.fetch = async url => (requested.push(url), {
    ok: url in responses,
    status: url in responses ? 200 : 404,
    json: async () => responses[url],
    blob: async () => responses[url],
  })
  return requested
}

const player = (video, events) => createReplayPlayer({
  video: () => video,
  onFrame: frame => events.push(['frame', frame.status.action]),
  onEpisode: (episode, replay) => events.push(['episode', episode.id, replay]),
  onEpisodeEnd: episode => events.push(['end', episode.id]),
  onWaiting: () => events.push(['waiting']),
  onLatest: () => {},
  onConnection: () => {},
})

test('H: every shown video frame carries the decision recorded for that frame', async () => {
  installBrowser({
    '/api/live/episode': LATEST,
    '/api/live/statuses?generation=g&episode=4&start=0': { statuses: STATUSES },
    '/api/live/video?generation=g&episode=4': 'video-4',
  })
  const video = fakeVideo()
  const events = []
  const replays = player(video, events)
  replays.start()
  await settle()
  video.frame(0)
  video.frame(1)

  assert.equal(video.src, 'blob:video-4')
  assert.deepEqual(events, [['episode', 4, false], ['frame', 'left'], ['frame', 'right']])
  replays.stop()
})

test('I: a finished attempt is followed by the best replays in a loop', async () => {
  const replay = { ...EPISODE, id: 2 }
  installBrowser({
    '/api/live/episode': { ...LATEST, replays: [replay] },
    '/api/live/statuses?generation=g&episode=4&start=0': { statuses: STATUSES },
    '/api/live/statuses?generation=g&episode=2&start=0': { statuses: STATUSES },
    '/api/live/video?generation=g&episode=4': 'video-4',
    '/api/live/video?generation=g&episode=2': 'video-2',
  })
  const video = fakeVideo()
  const events = []
  const replays = player(video, events)
  replays.start()
  await settle()
  video.end()
  await settle()
  video.end()
  await settle()

  assert.deepEqual(events, [['episode', 4, false], ['end', 4], ['episode', 2, true], ['end', 2], ['episode', 2, true]])
  assert.equal(video.src, 'blob:video-2')
  replays.stop()
})

test('J: an attempt that is gone before its video loaded is skipped while waiting', async () => {
  installBrowser({
    '/api/live/episode': LATEST,
    '/api/live/statuses?generation=g&episode=4&start=0': { statuses: STATUSES },
  })
  const video = fakeVideo()
  const events = []
  const replays = player(video, events)
  replays.start()
  await settle()

  assert.deepEqual(events, [['waiting']])
  assert.equal(video.played, 0)
  replays.stop()
})

test('K: the next video downloads while the current attempt still plays', async () => {
  const replay = { ...EPISODE, id: 2 }
  const requested = installBrowser({
    '/api/live/episode': { ...LATEST, replays: [replay] },
    '/api/live/statuses?generation=g&episode=4&start=0': { statuses: STATUSES },
    '/api/live/statuses?generation=g&episode=2&start=0': { statuses: STATUSES },
    '/api/live/video?generation=g&episode=4': 'video-4',
    '/api/live/video?generation=g&episode=2': 'video-2',
  })
  const video = fakeVideo()
  const replays = player(video, [])
  replays.start()
  await settle()
  const beforeEnd = [...requested]
  video.end()
  await settle()

  assert.ok(beforeEnd.includes('/api/live/video?generation=g&episode=2'))
  assert.equal(requested.filter(url => url === '/api/live/video?generation=g&episode=2').length, 1)
  assert.equal(video.src, 'blob:video-2')
  replays.stop()
})
