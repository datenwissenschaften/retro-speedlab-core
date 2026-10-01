import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  arrivalBanner, arrivalOutcome, createArrivalTracker, dangerNote, dangerTitle, holdsBest, inProgressLine, newlyMastered, recentAttempts, resultBanner, status,
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

test('a win outranks a new best, which outranks an improvement', () => {
  assert.equal(arrivalOutcome(episode(3, { won: true, new_best: true, score: 9 }), [1, 9]).kind, 'won')
  assert.equal(arrivalOutcome(episode(3, { new_best: true, score: 9 }), [1, 9]).kind, 'best')
  assert.deepEqual(arrivalOutcome(episode(3, { score: -24 }), [-27, -24]), { kind: 'improved', previous: -27 })
})

test('equal, lower or unaligned scores are plain attempts', () => {
  assert.equal(arrivalOutcome(episode(3, { score: -27 }), [-27, -27]).kind, 'attempt')
  assert.equal(arrivalOutcome(episode(3, { score: -30 }), [-27, -30]).kind, 'attempt')
  assert.equal(arrivalOutcome(episode(3, { score: -20 }), [-30, -25]).kind, 'attempt')
  assert.equal(arrivalOutcome(episode(3, { score: -20 }), undefined).kind, 'attempt')
  assert.equal(arrivalOutcome(episode(3, { score: -20 }), [-20]).kind, 'attempt')
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

test('mastery fires only when a known phase flips to mastered', () => {
  const learning = { Level1: { Play: { mastered: false, wins: 7, win_target: 8 } } }
  const mastered = { Level1: { Play: { mastered: true, wins: 8, win_target: 8 } } }
  assert.deepEqual(newlyMastered(learning, mastered), [{ savestate: 'Level1', phase: 'Play', phases: 1, wins: 8, winTarget: 8 }])
  assert.deepEqual(newlyMastered(mastered, mastered), [])
  assert.deepEqual(newlyMastered(undefined, mastered), [])
  assert.deepEqual(newlyMastered({ Level2: learning.Level1 }, mastered), [])
})

test('arrival banners name the attempt and its reward', () => {
  assert.deepEqual(arrivalBanner(episode(3, { score: -24 }), [-27, -24]), { kind: 'arrival', title: 'New attempt · #4', detail: 'reward -24.0 · up from -27.0' })
  assert.deepEqual(arrivalBanner(episode(3, { score: -30 }), [-27, -30]), { kind: 'arrival', title: 'New attempt · #4', detail: 'reward -30.0' })
  assert.equal(arrivalBanner(episode(3, { new_best: true, score: 5 }), [5]).title, '★ New best attempt')
  assert.equal(arrivalBanner(episode(3, { won: true, score: 5 }), [5]).title, '★ Attempt won')
})

test('result banners mark the attempt that holds the level best', () => {
  assert.deepEqual(resultBanner(episode(3, { score: -20 }), -20), { kind: 'best', title: 'Result · #4', detail: 'reward -20.0 · ★ best so far' })
  assert.deepEqual(resultBanner(episode(3, { score: -25 }), -20), { kind: 'result', title: 'Result · #4', detail: 'reward -25.0' })
  assert.equal(resultBanner(episode(3, { score: -25 }), null).kind, 'result')
  assert.equal(resultBanner(episode(3, { won: true, score: -25 }), -20).kind, 'won')
  assert.equal(holdsBest(episode(3, { score: -25 }), null), false)
})

const replaying = (attempt, level) => episode(attempt - 1, { attempt, level })

test('A: replaying #5 while #6 is played shows the replay and the next attempt in progress', () => {
  assert.equal(status(true, null, replaying(5, 'Level1')), 'Replay')
  assert.equal(inProgressLine(replaying(5, 'Level1'), { attempt: 6, level: 'Level1' }), 'Next attempt in progress')
})

test('A: a replay older than the newest completed attempt names the attempt in progress', () => {
  assert.equal(inProgressLine(replaying(5, 'Level1'), { attempt: 7, level: 'Level1' }), 'Attempt #7 in progress')
  assert.equal(inProgressLine(replaying(5, 'Level1'), { attempt: 2, level: 'Level2' }), 'Level 2 · Attempt #2 in progress')
})

test('B: a completed #6 replaces the replay once and is listed once', () => {
  const arrived = createArrivalTracker()
  const six = episode(5, { attempt: 6, score: -26 })
  assert.equal(arrived('g', six, false), true)
  assert.equal(status(true, { arrival: true }, six), 'New')
  assert.equal(status(true, null, six), 'Replay')
  const bars = recentAttempts(six, [-30, -28, -27, -29, -26], null, 5)
  assert.equal(bars.filter(bar => bar.attempt === 6).length, 1)
  assert.equal(bars[0].attempt, 6)
})

test('C: a new best is announced once however often the same attempt is polled or replayed', () => {
  const arrived = createArrivalTracker()
  const best = episode(5, { attempt: 6, score: -20, new_best: true })
  const announced = [1, 2, 3].filter(() => arrived('g', best, false)).map(() => arrivalBanner(best, [-26, -20]).title)
  assert.deepEqual(announced, ['★ New best attempt'])
  assert.equal(arrived('g', best, true), false)
  assert.equal(recentAttempts(best, [-26, -20], -20, 5)[0].best, true)
})

test('D: nothing in progress hides the in-progress line', () => {
  assert.equal(inProgressLine(replaying(5, 'Level1'), null), null)
  assert.equal(inProgressLine(null, null), null)
  assert.equal(status(false, null, replaying(5, 'Level1')), 'Offline')
  assert.equal(status(true, null, null), 'Waiting')
})

test('E: rewards are called rewards, never scores or points', () => {
  const texts = [arrivalBanner(episode(1, { score: -26 }), [-26]), resultBanner(episode(1, { score: -26 }), -20)]
    .flatMap(banner => [banner.title, banner.detail])
  assert.ok(texts.some(text => text.includes('reward -26.0')))
  assert.ok(texts.every(text => !/score|points/i.test(text)))
})

test('F: a game-defined win is called a win, never a level clear', () => {
  const won = episode(1, { won: true, score: 3 })
  for (const banner of [arrivalBanner(won, [3]), resultBanner(won, 3)]) {
    assert.equal(banner.kind, 'won')
    assert.ok(!/clear/i.test(`${banner.title} ${banner.detail}`))
  }
})

test('G: danger wording follows the real failure window and whether a place is known', () => {
  assert.equal(dangerTitle({ located: true, count: 6 }), 'Failed here 6×')
  assert.equal(dangerTitle({ located: false, count: 6 }), 'Failed 6×')
  assert.equal(dangerNote(6, false), 'From the last 6 failed attempts')
  assert.equal(dangerNote(100, true), 'Where the last 100 failed attempts ended')
  assert.equal(dangerNote(1, true), 'Where the last 1 failed attempt ended')
})
