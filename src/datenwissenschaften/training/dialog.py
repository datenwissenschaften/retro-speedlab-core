import threading
import time
from collections.abc import Callable

import httpx
from loguru import logger

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SECONDS_BETWEEN_LINES = 90.0
TIMEOUT_SECONDS = 20
MAX_TOKENS = 80
MAX_LINE_LENGTH = 160
VOICE = (
    "You are {persona}, a cheerful, slightly cocky AI gamer girl who is learning to play {game} live on a stream "
    "by trial and error. Answer with exactly one short spoken line (at most 15 words) about what just happened, "
    "in the first person, without quotes, hashtags or emojis."
)


class DialogWriter:
    def __init__(self, persona: str, game: str, models: tuple[str, ...], api_key: str) -> None:
        self.voice = VOICE.format(persona=persona, game=game)
        self.models = models
        self.api_key = api_key
        self.busy = False
        self.last_request = -SECONDS_BETWEEN_LINES

    def comment(self, situation: str, deliver: Callable[[str], None]) -> None:
        now = time.monotonic()
        if self.busy or now - self.last_request < SECONDS_BETWEEN_LINES:
            return
        self.busy, self.last_request = True, now
        threading.Thread(target=self._write, args=(situation, deliver), name="dialog", daemon=True).start()

    def _write(self, situation: str, deliver: Callable[[str], None]) -> None:
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "models": list(self.models),
                    "max_tokens": MAX_TOKENS,
                    "reasoning": {"enabled": False},
                    "messages": [{"role": "system", "content": self.voice}, {"role": "user", "content": situation}],
                },
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            line = str(response.json()["choices"][0]["message"]["content"]).strip()
            if line:
                deliver(line[:MAX_LINE_LENGTH])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            logger.warning(f"No dialog line from {', '.join(self.models)}: {error}")
        finally:
            self.busy = False
