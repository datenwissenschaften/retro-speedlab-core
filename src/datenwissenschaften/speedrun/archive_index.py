import heapq
import random
from bisect import bisect_left, insort

from datenwissenschaften.speedrun.runner import CellKey
from datenwissenschaften.speedrun.search import EXPLOIT, LEADING_SHARE, Cell

Lead = tuple[float, int, CellKey]
Rank = tuple[float, int, int, CellKey]
STALE_RANKS = 4


class ArchiveIndex:
    def __init__(self, cells: dict[CellKey, Cell]) -> None:
        self.cells = cells
        self.order = {key: index for index, key in enumerate(cells)}
        self.added = len(cells)
        self.weights = {key: cell.weight for key, cell in cells.items()}
        self.leading = sorted(self.lead(key) for key in cells)
        self.ranks: list[Rank] = []
        self.rerank()

    def lead(self, key: CellKey) -> Lead:
        return self.cells[key].score, self.order[key], key

    def rank(self, key: CellKey) -> Rank:
        cell = self.cells[key]
        return cell.score, -cell.visits, self.order[key], key

    def rerank(self) -> None:
        self.ranks = [self.rank(key) for key, cell in self.cells.items() if cell.trail is not None]
        heapq.heapify(self.ranks)

    def ranked(self, key: CellKey) -> None:
        if self.cells[key].trail is not None:
            heapq.heappush(self.ranks, self.rank(key))
        if len(self.ranks) > STALE_RANKS * len(self.cells):
            self.rerank()

    def unlead(self, key: CellKey) -> None:
        del self.leading[bisect_left(self.leading, self.lead(key))]

    def current(self, rank: Rank) -> bool:
        key = rank[-1]
        return key in self.cells and self.cells[key].trail is not None and self.rank(key) == rank

    def put(self, key: CellKey, cell: Cell) -> None:
        if key in self.cells:
            self.unlead(key)
        else:
            self.order[key], self.added = self.added, self.added + 1
        self.cells[key] = cell
        self.weights[key] = cell.weight
        insort(self.leading, self.lead(key))
        self.ranked(key)

    def evict(self) -> None:
        rank = heapq.heappop(self.ranks)
        while not self.current(rank):
            rank = heapq.heappop(self.ranks)
        key = rank[-1]
        self.unlead(key)
        del self.cells[key], self.weights[key], self.order[key]

    def reweigh(self) -> None:
        self.weights = {key: cell.weight for key, cell in self.cells.items()}

    def choose(self, rng: random.Random) -> Cell:
        if rng.random() < EXPLOIT:
            keys = [key for _, _, key in self.leading[-max(1, len(self.cells) // LEADING_SHARE) :]]
            return self.visit(rng.choices(keys, weights=[self.weights[key] for key in keys])[0])
        return self.visit(rng.choices(list(self.weights), weights=list(self.weights.values()))[0])

    def visit(self, key: CellKey) -> Cell:
        cell = self.cells[key]
        cell.visits += 1
        self.weights[key] = cell.weight
        self.ranked(key)
        return cell
