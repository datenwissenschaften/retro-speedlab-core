import assert from 'node:assert/strict'
import { test } from 'node:test'
import { levelOf, pathNodes, splitDelta, splitTime } from './progressPath.js'

const phase = mastered => ({ mastered, wins: mastered ? 8 : 2, win_target: 8 })
const LEVELS = { 'Level 1': ['Play', 'Grow', 'Door'], 'Level 2': ['Level2', 'Island2'] }

test('a level with every state mastered is one node, the others show their states', () => {
  const curriculum = {
    Menu: phase(true), Play: phase(true), Grow: phase(true), Door: phase(true), 'Level 1': phase(false),
    Level2: phase(true), Island2: phase(false), 'Level 2': phase(false),
  }
  assert.deepEqual(pathNodes(curriculum, LEVELS).map(node => node.name), ['Menu', 'Level 1', 'Level2', 'Island2'])
  assert.equal(pathNodes(curriculum, LEVELS)[1].level, true)
})

test('before its states are mastered a level is not shown as a node', () => {
  const curriculum = { Menu: phase(true), Play: phase(true), Grow: phase(false), Door: phase(false), 'Level 1': phase(false) }
  assert.deepEqual(pathNodes(curriculum, LEVELS).map(node => node.name), ['Menu', 'Play', 'Grow', 'Door'])
})

test('split times read like a speedrun timer, deltas against the best run', () => {
  assert.equal(splitTime(72.349), '1:12.34')
  assert.equal(splitTime(9.5), '0:09.50')
  assert.equal(splitTime(null), '—')
  assert.deepEqual(splitDelta(73.15, 72.35), { text: '+0.80', ahead: false })
  assert.deepEqual(splitDelta(70, 72.35), { text: '−2.35', ahead: true })
  assert.equal(splitDelta(null, 72.35), null)
})

test('the level shown is the level of the state being played', () => {
  assert.equal(levelOf('Grow', LEVELS), 'Level 1')
  assert.equal(levelOf('Island2', LEVELS), 'Level 2')
  assert.equal(levelOf('Menu', LEVELS), 'Menu')
  assert.equal(levelOf(undefined, LEVELS), '—')
})
