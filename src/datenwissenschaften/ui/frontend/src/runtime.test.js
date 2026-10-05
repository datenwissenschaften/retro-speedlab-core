import assert from 'node:assert/strict'
import { test } from 'node:test'
import { elapsed, minutesLeft } from './runtime.js'

const start = '2026-10-01T08:00:00.000+00:00'
const at = iso => new Date(iso)

test('elapsed time reads in minutes, hours and days', () => {
  assert.equal(elapsed(start, at('2026-10-01T08:00:59Z')), '0m')
  assert.equal(elapsed(start, at('2026-10-01T08:42:00Z')), '42m')
  assert.equal(elapsed(start, at('2026-10-01T14:42:30Z')), '6h 42m')
  assert.equal(elapsed(start, at('2026-10-02T08:00:00Z')), '1d 0h')
  assert.equal(elapsed(start, at('2026-10-04T15:59:00Z')), '3d 7h')
})

test('a clock behind the server never shows negative time', () => {
  assert.equal(elapsed(start, at('2026-10-01T07:59:00Z')), '0m')
})

test('a lab run counts down whole minutes and never shows zero', () => {
  assert.equal(minutesLeft('2026-10-01T09:00:00Z', at('2026-10-01T08:15:30Z')), 45)
  assert.equal(minutesLeft('2026-10-01T09:00:00Z', at('2026-10-01T08:59:50Z')), 1)
  assert.equal(minutesLeft('2026-10-01T09:00:00Z', at('2026-10-01T09:05:00Z')), 1)
})
