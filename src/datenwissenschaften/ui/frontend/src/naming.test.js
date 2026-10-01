import assert from 'node:assert/strict'
import { test } from 'node:test'
import { gameTitle, words } from './naming.js'

test('game ids lose their platform suffix and read as words', () => {
  assert.equal(gameTitle('SnakeRattleNRoll-Nes-v0'), "Snake Rattle 'n' Roll")
  assert.equal(gameTitle('Airstriker-Genesis-v0'), 'Airstriker')
  assert.equal(gameTitle('SuperMarioBros3-Nes-v0'), 'Super Mario Bros 3')
})

test('savestate and phase names read as words', () => {
  assert.equal(words('Level1'), 'Level 1')
  assert.equal(words('Level1Boss'), 'Level 1 Boss')
  assert.equal(words('FindTheScale'), 'Find The Scale')
  assert.equal(words('Play'), 'Play')
})
