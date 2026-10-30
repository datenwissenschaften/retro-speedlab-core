# Ideas for later

Taken from what makes Gemini Plays Pokémon engaging, mapped to Retro Speedlab. Not planned yet.

## Viewer interaction
- **Ask the AI (channel points):** viewers redeem "Ask the AI". A separate Claude call (OpenRouter BYOK) answers in chat from the live snapshot, lab reports and notes. It never influences the run and keeps no memory between questions.
- **Twitch predictions:** "Does the AI beat Level 2 within the hour?" opened and resolved automatically from milestones (Helix Predictions API).
- **Viewer-named seeds:** a redeem names one search worker for an hour, and the stream shows "Worker *name* found the way out".

## Transparency, like their GitHub tracker
- **Public tracker page:** the game package's notes (`NOTES.md`, `notes/level*.md`), the lab reports, the current goals and the lab's newest commits, live on the website.
- **Movie download:** the verified `.bk2` and its frame count, so the TAS community can replay it in BizHawk. Submit to TASVideos once it beats a publication.
- **Search turn lifecycle panel:** a "How it works" panel in the style of their agent turn lifecycle: pick a cell → load the savestate → try inputs → keep the new cells → leave the state → hold check → verify from power-on.
- **Filmstrip:** a contact sheet of key frames of the newest segment, with frame index and the buttons held.

## Overlay and story
- **Avatar moods:** the AI avatar's pose follows the run: focused while searching, frustrated when stuck, celebrating a PB.
- **TAS ghost:** replay Aglar's TASVideos movie in our emulator next to the AI's run, if it syncs on stable-retro. It needs an emulator check first.
- **Split comparison video:** after each improvement, a side-by-side of the old and the new segment, with the frames saved.
- **Update announcements:** when a new release loads, a banner such as "New game package 2026.10.29-25: endings must now hold". `deploy.sh` would take a one-line release note and publish it next to `server.release`, and the stream would show it for a minute after the reload.
- **Daily recap:** a short clip and post at the end of the day: time saved, levels reached, lab fixes shipped.

## Engine roadmap
- **Reactivate the route maps (engine only, never shown on the stream):** the distance-to-goal maps per state (`models/<game>/<state>/route.json`) came from the old training and are gone, so `runner.remaining()` is always `None`. Build them again from the search itself, for example from the positions of the fastest crossings of each piece, so the search archive can weight cells by distance again and checkpoints can use the distance to the goal instead of the state's reward. They stay out of the frontend.
- **Less game knowledge (like their "reduce harness assist"):** Claude derives the game states from RAM and published routes on its own, so a new game needs only a ROM, a RAM map and the records.
- **New game support:** more NES games, each with its own leaderboard sources; a games hub on the site and rotation on the stream.
- **Any% with warps:** a second category per game with its own movie, splits and records (TASVideos #122 for Snake Rattle 'n Roll).
- **Head-to-head:** two search strategies, or two Claude models in the lab, racing on the same game.
- **Reflection and critique, like their every-N-turn reviews:** an independent Claude review of the last lab runs, every few runs, written into the notes.
