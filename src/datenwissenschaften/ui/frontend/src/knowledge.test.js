import assert from 'node:assert/strict'
import { test } from 'node:test'
import { layaInputs } from './knowledge.js'

const knowledge = {
  checkpoint: 'convaiinnovations/laya',
  states: [
    { name: 'Menu', seeded: false, demonstrations: 138 },
    { name: 'Eat', seeded: true, demonstrations: 1200 },
    { name: 'Scale', seeded: true, demonstrations: 0 },
  ],
}

const value = (rows, name) => rows.find(row => row.name === name)?.value

test('nothing is claimed before training published what Laya learns from', () => {
  assert.deepEqual(layaInputs(undefined, {}, {}, 10), [])
})

test('Laya’s inputs count seeded states and demonstrated moves', () => {
  const rows = layaInputs(knowledge, { imitation_loss: 1.234 }, {}, 10)
  assert.equal(value(rows, 'Pretrained model'), 'convaiinnovations/laya')
  assert.equal(value(rows, 'Start points'), '2 of 3 states seeded from power-on')
  assert.equal(value(rows, 'Demonstrations'), '1,338 moves in 2 states')
  assert.equal(value(rows, 'Imitation loss'), '1.23')
})

test('without demonstrations or advice there is no imitation loss to show', () => {
  const rows = layaInputs({ ...knowledge, states: knowledge.states.map(state => ({ ...state, demonstrations: 0 })) }, { imitation_loss: 0 }, {}, 10)
  assert.equal(value(rows, 'Demonstrations'), 'None yet')
  assert.equal(value(rows, 'Imitation loss'), undefined)
})

test('the advisor row counts practice moves across states on the practice emulators', () => {
  const advisors = { Play: { num_timesteps: 98000 }, Heavy: { num_timesteps: 4100 } }
  assert.equal(value(layaInputs(knowledge, {}, advisors, 10), 'Advisor'), '102,100 practice moves on 9 emulators')
  assert.equal(value(layaInputs(knowledge, {}, {}, 10), 'Advisor'), '9 practice emulators warming up')
})
