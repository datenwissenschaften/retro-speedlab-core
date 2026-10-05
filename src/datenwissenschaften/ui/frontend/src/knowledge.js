import { fmt } from './naming.js'

export const LAB_SOURCES = [
  { name: 'Demonstrations', detail: 'Play recorded from power-on: people and the lab’s own successful runs' },
  { name: 'Videos', detail: 'Speedruns, tool-assisted runs and longplays: the route, reproduced and recorded by the lab' },
  { name: 'Guides and RAM maps', detail: 'Walkthroughs, manuals and Data Crystal' },
]

export const layaInputs = (knowledge, laya) => {
  if (!knowledge) return []
  const { states } = knowledge
  const seeded = states.filter(state => state.seeded).length
  const moves = states.reduce((total, state) => total + state.demonstrations, 0)
  const demonstrated = states.filter(state => state.demonstrations > 0).length
  return [
    { name: 'Pretrained model', value: knowledge.checkpoint },
    { name: 'Questions', value: `${states.length} states, one goal each` },
    { name: 'Rewards', value: 'Written by the lab for every state' },
    { name: 'Start points', value: `${seeded} of ${states.length} states seeded from power-on` },
    { name: 'Demonstrations', value: moves ? `${fmt(moves)} moves in ${demonstrated} ${demonstrated === 1 ? 'state' : 'states'}` : 'None yet' },
    ...(moves && laya?.imitation_loss != null ? [{ name: 'Imitation loss', value: fmt(laya.imitation_loss, 2) }] : []),
  ]
}
