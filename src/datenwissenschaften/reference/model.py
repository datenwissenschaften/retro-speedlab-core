from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from torch import nn

from datenwissenschaften.reference.samples import Samples

EPOCHS = 12
BATCH = 256
LEARNING_RATE = 1e-3
VALIDATION_SHARE = 0.1
SHIFT_PIXELS = 4
COLOUR_JITTER = 0.2
COLOUR_OFFSET = 0.12
NOISE = 0.03
COORDINATE_SCALE = 256.0
PLAYING = 0.5


@dataclass(slots=True, frozen=True)
class Reading:
    playing: np.ndarray
    areas: np.ndarray
    coordinates: np.ndarray


class PositionNet(nn.Module):
    def __init__(self, areas: int, height: int, width: int) -> None:
        super().__init__()
        channels = (3, 32, 64, 128, 128)
        layers: list[nn.Module] = []
        for inputs, outputs in zip(channels, channels[1:], strict=False):
            layers += [nn.Conv2d(inputs, outputs, 3, stride=2, padding=1), nn.BatchNorm2d(outputs), nn.ReLU()]
        self.features = nn.Sequential(*layers, nn.Flatten())
        flat = self.features(torch.zeros(1, 3, height, width)).shape[1]
        self.body = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Dropout(0.2))
        self.playing = nn.Linear(512, 1)
        self.area = nn.Linear(512, areas)
        self.coordinates = nn.Linear(512, 3)

    def forward(self, screens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body = self.body(self.features(screens))
        return self.playing(body).squeeze(1), self.area(body), self.coordinates(body)


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("The guide trains on a GPU, and torch sees none: run the app with --gpus all")
    return torch.device("cuda")


def as_input(screens: np.ndarray, gpu: torch.device) -> torch.Tensor:
    return torch.from_numpy(screens).to(gpu).permute(0, 3, 1, 2).float() / 255


def jitter(screens: torch.Tensor, spread: float) -> torch.Tensor:
    return spread * (2 * torch.rand(screens.shape[0], 3, 1, 1, device=screens.device) - 1)


def augmented(screens: torch.Tensor) -> torch.Tensor:
    shift = [int(value) for value in torch.randint(-SHIFT_PIXELS, SHIFT_PIXELS + 1, (2,))]
    coloured = screens * (1 + jitter(screens, COLOUR_JITTER)) + jitter(screens, COLOUR_OFFSET)
    noisy = coloured + NOISE * torch.randn_like(screens)
    return torch.roll(noisy, shifts=shift, dims=(2, 3)).clamp(0, 1)


def losses(net: PositionNet, screens: torch.Tensor, labels: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    playing, areas, coordinates = net(screens)
    inside = labels[:, 0] >= 0
    loss = nn.functional.binary_cross_entropy_with_logits(playing, inside.float())
    if inside.any():
        loss = loss + nn.functional.cross_entropy(areas[inside], classes[inside])
        loss = loss + nn.functional.smooth_l1_loss(coordinates[inside], labels[inside, 1:] / COORDINATE_SCALE)
    return loss


def evaluate(net: PositionNet, screens: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> str:
    reading = read_positions(net, list(range(int(classes.max()) + 1)), screens)
    inside = labels[:, 0] >= 0
    seen = (reading.playing == inside).mean()
    area = (reading.areas[inside] == classes[inside]).mean()
    error = np.abs(reading.coordinates[inside] - labels[inside, 1:]).mean(0).round(1)
    return f"in play {seen:.3f}, area {area:.3f}, mean error x/y/height {error.tolist()}"


def train(samples: Samples) -> tuple[PositionNet, list[int]]:
    gpu = device()
    screens, labels = np.stack(samples.screens), np.asarray(samples.labels, dtype=np.float32)
    known = sorted({int(area) for area in labels[:, 0] if area >= 0})
    classes = np.asarray([known.index(int(area)) if area >= 0 else 0 for area in labels[:, 0]])
    order = np.random.default_rng(0).permutation(len(labels))
    held, fitted = order[: int(len(order) * VALIDATION_SHARE)], order[int(len(order) * VALIDATION_SHARE) :]
    net = PositionNet(len(known), screens.shape[1], screens.shape[2]).to(gpu)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        net.train()
        np.random.default_rng(epoch).shuffle(fitted)
        for start in range(0, len(fitted), BATCH):
            batch = fitted[start : start + BATCH]
            truth = torch.from_numpy(labels[batch]).to(gpu)
            optimizer.zero_grad()
            losses(
                net, augmented(as_input(screens[batch], gpu)), truth, torch.from_numpy(classes[batch]).to(gpu)
            ).backward()
            optimizer.step()
        logger.info(f"Guide epoch {epoch + 1}/{EPOCHS}: {evaluate(net, screens[held], labels[held], classes[held])}")
    return net, known


def read_positions(net: PositionNet, known: list[int], screens: np.ndarray) -> Reading:
    gpu = device()
    net.eval()
    readings: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, len(screens), BATCH):
            playing, areas, coordinates = net(as_input(screens[start : start + BATCH], gpu))
            readings.append((torch.sigmoid(playing) > PLAYING, areas.argmax(1), coordinates * COORDINATE_SCALE))
    playing, areas, coordinates = (torch.cat(parts).cpu().numpy() for parts in zip(*readings, strict=True))
    return Reading(playing, np.asarray(known)[areas], np.rint(coordinates).astype(int))
