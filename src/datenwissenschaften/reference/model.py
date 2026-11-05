from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from loguru import logger
from torch import nn

from datenwissenschaften.gpu import gpu
from datenwissenschaften.reference.familiar import nearest
from datenwissenschaften.reference.samples import NOT_PLAYING, Samples

EPOCHS = 12
BATCH = 256
LEARNING_RATE = 1e-3
VALIDATION_SHARE = 0.1
SHIFT_PIXELS = 4
COLOUR_JITTER = 0.2
COLOUR_OFFSET = 0.12
NOISE = 0.03
PAGE = 256
REACH_QUANTILE = 0.99

Place = tuple[int, int]
Split = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True, frozen=True)
class Reading:
    places: np.ndarray
    coordinates: np.ndarray
    embeddings: torch.Tensor


class PositionNet(nn.Module):
    def __init__(self, places: int, height: int, width: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        channels = (3, 32, 64, 128, 128)
        layers: list[nn.Module] = []
        for inputs, outputs in zip(channels, channels[1:], strict=False):
            layers += [nn.Conv2d(inputs, outputs, 3, stride=2, padding=1), nn.BatchNorm2d(outputs), nn.ReLU()]
        self.features = nn.Sequential(*layers, nn.Flatten())
        flat = self.features(torch.zeros(1, 3, height, width)).shape[1]
        self.body = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Dropout(0.2))
        self.place = nn.Linear(512, places)
        self.coordinates = nn.Linear(512, 2)

    def forward(self, screens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body = self.body(self.features(screens))
        return self.place(body), self.coordinates(body), nn.functional.normalize(body, dim=1)


def device() -> torch.device:
    return gpu()


def as_input(screens: np.ndarray, gpu: torch.device) -> torch.Tensor:
    return torch.as_tensor(screens).to(gpu).permute(0, 3, 1, 2).float() / 255


def jitter(screens: torch.Tensor, spread: float) -> torch.Tensor:
    return spread * (2 * torch.rand(screens.shape[0], 3, 1, 1, device=screens.device) - 1)


def augmented(screens: torch.Tensor) -> torch.Tensor:
    shift = [int(value) for value in torch.randint(-SHIFT_PIXELS, SHIFT_PIXELS + 1, (2,))]
    coloured = screens * (1 + jitter(screens, COLOUR_JITTER)) + jitter(screens, COLOUR_OFFSET)
    noisy = coloured + NOISE * torch.randn_like(screens)
    return torch.roll(noisy, shifts=shift, dims=(2, 3)).clamp(0, 1)


@dataclass(slots=True, frozen=True)
class Trained:
    net: PositionNet
    places: list[Place]
    allowed: np.ndarray
    seen: torch.Tensor
    reach: float


def in_page(labels: np.ndarray) -> np.ndarray:
    return np.stack([labels[:, 1], labels[:, 2] % PAGE], 1)


def place_of(label: np.ndarray) -> Place:
    return int(label[0]), int(label[2]) // PAGE


def losses(net: PositionNet, screens: torch.Tensor, coordinates: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    places, predicted, _ = net(screens)
    return nn.functional.cross_entropy(places, classes) + nn.functional.smooth_l1_loss(predicted, coordinates / PAGE)


def evaluate(net: PositionNet, screens: np.ndarray, coordinates: np.ndarray, classes: np.ndarray) -> str:
    reading = read_positions(net, screens)
    place = (reading.places.argmax(1) == classes).mean()
    error = np.median(np.abs(reading.coordinates - coordinates), 0).tolist()
    return f"place {place:.3f}, median error x/y {error}"


def allowed_places(levels: np.ndarray, classes: np.ndarray, level_count: int, places: int) -> np.ndarray:
    allowed = np.zeros((level_count, places), dtype=bool)
    allowed[levels, classes] = True
    return allowed


def split(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.as_tensor(np.random.default_rng(0).permutation(count), device=device())
    return order[: int(count * VALIDATION_SHARE)], order[int(count * VALIDATION_SHARE) :]


def fit(net: PositionNet, screens: np.ndarray, coordinates: np.ndarray, classes: np.ndarray, parts: Split) -> None:
    gpu = device()
    held, fitted = (part.cpu().numpy() for part in parts)
    optimizer = cast(torch.optim.Optimizer, torch.optim.Adam(net.parameters(), lr=LEARNING_RATE))
    for epoch in range(EPOCHS):
        net.train()
        np.random.default_rng(epoch).shuffle(fitted)
        for start in range(0, len(fitted), BATCH):
            batch = fitted[start : start + BATCH]
            optimizer.zero_grad()
            truth = torch.as_tensor(coordinates[batch], dtype=torch.float32).to(gpu)
            classified = torch.as_tensor(classes[batch]).to(gpu)
            torch.autograd.backward(losses(net, augmented(as_input(screens[batch], gpu)), truth, classified))
            optimizer.step()
        logger.info(
            f"Guide epoch {epoch + 1}/{EPOCHS}: {evaluate(net, screens[held], coordinates[held], classes[held])}"
        )


def train(samples: Samples, level_count: int) -> Trained:
    playing = [index for index, label in enumerate(samples.labels) if label[0] != NOT_PLAYING]
    screens, labels = np.stack([samples.screens[index] for index in playing]), np.asarray(samples.labels)[playing]
    known = sorted({place_of(label) for label in labels})
    classes, coordinates = np.asarray([known.index(place_of(label)) for label in labels]), in_page(labels)
    net = PositionNet(len(known), screens.shape[1], screens.shape[2]).to(device())
    held, fitted = split(len(labels))
    fit(net, screens, coordinates, classes, (held, fitted))
    seen = read_positions(net, screens).embeddings
    reach = float(np.quantile(nearest(seen[held], seen[fitted]), REACH_QUANTILE))
    logger.info(f"The guide trusts video screens within {reach:.3f} of a screen it trained on")
    return Trained(net, known, allowed_places(labels[:, 3], classes, level_count, len(known)), seen, reach)


def read_positions(net: PositionNet, screens: np.ndarray) -> Reading:
    net.eval()
    readings: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, len(screens), BATCH):
            places, coordinates, embeddings = net(as_input(screens[start : start + BATCH], device()))
            readings.append((torch.log_softmax(places, 1), coordinates * PAGE, embeddings))
    places, coordinates, embeddings = (torch.cat(parts) for parts in zip(*readings, strict=True))
    return Reading(places.cpu().numpy(), np.rint(coordinates.cpu().numpy()).astype(int), embeddings)


def familiar(trained: Trained, reading: Reading) -> np.ndarray:
    return nearest(reading.embeddings, trained.seen) <= trained.reach
