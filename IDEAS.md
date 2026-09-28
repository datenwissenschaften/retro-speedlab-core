# Stream ideas

Ideas for the live stream view that are not built yet.

## Plain-language thought bubble

Replace the percentage bars with a sentence that carries Laya's confidence as tone:
"I'm pretty sure: up-right, toward the food" or "No idea… let's try jumping". Show a short
"changed my mind" moment when the chosen move flips between consecutive decisions, so viewers
see the decision process instead of numbers. The data is already in every replay frame
(`probabilities`, `action`, and the `move` hint of the state).

## Update announcements

When a new release is loaded, show a banner such as "New brain update: Laya Model 2026.09.28-2 —
now knows where the snake really is". The deploy script already stamps every deploy with a
CalVer release; it would additionally take a one-line release note, publish it next to
`ui.release`, and the stream would show it for the first minute after the reload.
