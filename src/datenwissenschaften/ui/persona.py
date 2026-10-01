import hashlib

TAG_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
TAG_LENGTH = 6


def persona_tag(release: str) -> str:
    number = int.from_bytes(hashlib.sha256(release.encode("utf-8")).digest(), "big")
    tag = ""
    for _ in range(TAG_LENGTH):
        number, digit = divmod(number, len(TAG_ALPHABET))
        tag += TAG_ALPHABET[digit]
    return tag
