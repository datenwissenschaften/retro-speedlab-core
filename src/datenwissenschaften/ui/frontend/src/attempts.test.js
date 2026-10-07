import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  createArrivalTracker, dangerNote, dangerTitle, holdsBest, playingLine, recentAttempts, status,
} from './attempts.js'

const episode = (id, result) => ({ id, result: { score: 0, won: false, new_best: false, attempt: id + 1, level: 'Level1', ...result } })

test('a new completed attempt arrives once, replay loops do not repeat it', () => {
  const arrived = createArrivalTracker()
  assert.equal(arrived('g1', episode(4), false), true)
  assert.equal(arrived('g1', episode(4), true), false)
  assert.equal(arrived('g1', episode(4), false), false)
  assert.equal(arrived('g1', episode(5), true), false)
  assert.equal(arrived('g1', episode(5), false), true)
})

test('a restarted generation reusing an episode id is a new arrival', () => {
  const arrived = createArrivalTracker()
  assert.equal(arrived('g1', episode(0), false), true)
  assert.equal(arrived('g2', episode(0), false), true)
})

test('recent attempts list newest first between the worst recent score and the level best', () => {
  const bars = recentAttempts(episode(9, { score: -20 }), [-40, -30, -20], -10, 2)
  assert.deepEqual(bars.map(bar => bar.attempt), [10, 9])
  assert.deepEqual(bars.map(bar => bar.score), [-20, -30])
  assert.ok(bars[0].width > bars[1].width)
  assert.ok(bars.every(bar => bar.width > 0 && bar.width < 1))
  assert.ok(bars.every(bar => !bar.best))
})

test('the attempt that holds the level best is marked', () => {
  const bars = recentAttempts(episode(2, { score: -20 }), [-20, -20], -20, 5)
  assert.deepEqual(bars.map(bar => [bar.best, bar.width]), [[true, 1], [true, 1]])
})

test('recent attempts need scores that belong to the shown attempt', () => {
  assert.deepEqual(recentAttempts(null, [-1], null, 5), [])
  assert.deepEqual(recentAttempts(episode(1, { score: -5 }), [], null, 5), [])
  assert.deepEqual(recentAttempts(episode(1, { score: -5 }), [-3, -4], null, 5), [])
  assert.deepEqual(recentAttempts(episode(1, { score: -5 }), [-5], null, 5).map(bar => bar.width), [1])
})

test('an attempt holds the level best when it reaches it', () => {
  assert.equal(holdsBest(episode(3, { score: -20 }), -20), true)
  assert.equal(holdsBest(episode(3, { score: -25 }), -20), false)
  assert.equal(holdsBest(episode(3, { score: -25 }), null), false)
})

const replaying = (attempt, level) => episode(attempt - 1, { attempt, level })

test('A: replaying #5 shows the replay status', () => {
  assert.equal(status(true, replaying(5, 'Level1')), 'Replay')
})

test('B: a completed #6 replaces the replay once and is listed once', () => {
  const arrived = createArrivalTracker()
  const six = episode(5, { attempt: 6, score: -26 })
  assert.equal(arrived('g', six, false), true)
  assert.equal(status(true, six), 'Replay')
  const bars = recentAttempts(six, [-30, -28, -27, -29, -26], null, 5)
  assert.equal(bars.filter(bar => bar.attempt === 6).length, 1)
  assert.equal(bars[0].attempt, 6)
})

test('C: a new best arrives once however often the same attempt is polled or replayed', () => {
  const arrived = createArrivalTracker()
  const best = episode(5, { attempt: 6, score: -20, new_best: true })
  assert.deepEqual([1, 2, 3].map(() => arrived('g', best, false)), [true, false, false])
  assert.equal(arrived('g', best, true), false)
  assert.equal(recentAttempts(best, [-26, -20], -20, 5)[0].best, true)
})

test('D: the status shows offline and waiting', () => {
  assert.equal(status(false, replaying(5, 'Level1')), 'Offline')
  assert.equal(status(true, null), 'Waiting')
})

test('G: danger wording follows the real failure window and whether a place is known', () => {
  assert.equal(dangerTitle({ located: true, count: 6 }), 'Failed here 6×')
  assert.equal(dangerTitle({ located: false, count: 6 }), 'Failed 6×')
  assert.equal(dangerNote(6, false), 'From the last 6 failed attempts')
  assert.equal(dangerNote(100, true), 'Where the last 100 failed attempts ended')
  assert.equal(dangerNote(1, true), 'Where the last 1 failed attempt ended')
})

test('K: the curriculum shown is the state being played, and a full run says so', () => {
  assert.equal(playingLine('Play', episode(1, { curriculum: 'Feed', full_run: false })), 'Play')
  assert.equal(playingLine('Grow', episode(2, { curriculum: 'Menu', full_run: true })), 'Grow · full run')
  assert.equal(playingLine(undefined, episode(3, {})), '—')
})
