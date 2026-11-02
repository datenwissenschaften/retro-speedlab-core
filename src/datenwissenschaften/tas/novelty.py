import numpy as np

BYTE_VALUES = 256


class RamNovelty:
    def __init__(self, memory: np.ndarray) -> None:
        self.addresses = np.arange(len(memory))
        self.seen = np.zeros((len(memory), BYTE_VALUES), dtype=bool)
        self.see(memory)

    def see(self, memory: np.ndarray) -> int:
        fresh = ~self.seen[self.addresses, memory]
        self.seen[self.addresses, memory] = True
        return int(fresh.sum())
