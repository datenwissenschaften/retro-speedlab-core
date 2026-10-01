import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createReplayPlayer } from './replayPlayer.js'

const FRAME_STATUSES = [
  { attempt: 5, action: 'left', probabilities: { left: 0.7, right: 0.3 } },
  { attempt: 5, action: 'right', probabilities: { left: 0.2, right: 0.8 } },
]
const EPISODE = { id: 4, frame_rate: 1, frame_count: FRAME_STATUSES.length, result: { attempt: 5, level: 'Level1' } }
const RESPONSES = {
  '/api/live/episode': { generation: 'g', episode: EPISODE, replays: [], in_progress: { attempt: 6, level: 'Level1' }, summary: {} },
  '/api/live/frames?generation=g&episode=4&start=0': { frames: FRAME_STATUSES.map(status => ({ image: 'AA==', status })) },
}

const settle = () => new Promise(resolve => { setTimeout(resolve, 0) })

const installBrowser = () => {
  const frames = []
  globalThis.window = { setInterval: () => 0, clearInterval: () => {} }
  globalThis.fetch = async url => ({ ok: true, json: async () => RESPONSES[url] })
  globalThis.requestAnimationFrame = callback => { frames.push(callback) }
  globalThis.cancelAnimationFrame = () => {}
  globalThis.createImageBitmap = async () => ({ close: () => {} })
  return time => frames.shift()(time)
}

test('H: every drawn frame carries the decision recorded for that frame of the replayed attempt', async () => {
  const tick = installBrowser()
  const drawn = []
  const latest = []
  const player = createReplayPlayer({
    onFrame: frame => drawn.push(frame.status),
    onEpisode: () => {},
    onEpisodeEnd: () => {},
    onWaiting: () => {},
    onLatest: (episode, summary, inProgress) => latest.push(inProgress),
    onConnection: () => {},
  })
  player.start()
  await settle()
  tick(0)
  await settle()
  tick(0)
  tick(1)
  tick(1000)

  assert.deepEqual(drawn, FRAME_STATUSES)
  assert.deepEqual(latest, [{ attempt: 6, level: 'Level1' }])
  player.stop()
})
