import httpx
from loguru import logger

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.ui.relay import relay_key

TIMEOUT_SECONDS = 30.0
CLAIM_PATH = "/lab-requests/claim"


def claim_viewer_request(settings: RetroSpeedlabConfig) -> str | None:
    if not settings.ui.enabled:
        return None
    headers = {"X-API-Key": relay_key(settings.upload)}
    with httpx.Client(base_url=settings.upload.url, headers=headers, timeout=TIMEOUT_SECONDS) as client:
        try:
            response = client.post(CLAIM_PATH)
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning(f"No viewer lab request was claimed: {error}")
            return None
    request = response.json()
    return None if request is None else request["viewer"]
