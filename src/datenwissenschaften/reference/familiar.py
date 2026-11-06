import numpy as np
import torch

QUERY_BATCH = 256


def nearest(queries: torch.Tensor, seen: torch.Tensor) -> np.ndarray:
    return (
        torch.cat(
            [
                torch.cdist(queries[start : start + QUERY_BATCH], seen).min(1).values
                for start in range(0, len(queries), QUERY_BATCH)
            ]
        )
        .cpu()
        .numpy()
    )
