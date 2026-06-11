"""Nahual server for DINOv3.

Loads a DINOv3 backbone via the local `hubconf.py` (torch.hub.load with
`source='local'`), then runs it through the standard Nahual setup/process loop.

Run with:
    python server.py tcp://0.0.0.0:5123
"""

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse

import numpy

# Repo root containing hubconf.py — defaults to the directory of this file.
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", os.path.dirname(os.path.abspath(__file__)))


def _parse_args():
    parser = argparse.ArgumentParser(description="Nahual TCP server for DINOv3.")
    parser.add_argument(
        "address",
        nargs="?",
        default="tcp://0.0.0.0:5123",
        help="pynng listen address, e.g. tcp://0.0.0.0:5123",
    )
    return parser.parse_args()


def _torch_device(device: Optional[int]):
    import torch

    if torch.cuda.is_available():
        idx = 0 if device is None else int(device)
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def _extract_numpy_features(result) -> numpy.ndarray:
    if isinstance(result, dict):
        for key in ("x_norm_clstoken", "cls", "cls_token", "logits"):
            if key in result:
                result = result[key]
                break
        else:
            raise ValueError(
                f"DINOv3 output dict does not contain a supported feature key: {sorted(result)}"
            )
    elif isinstance(result, (list, tuple)):
        result = result[0]

    if hasattr(result, "detach"):
        result = result.float().detach().cpu().numpy()
    result = numpy.asarray(result)
    if result.ndim != 2:
        result = result.reshape(result.shape[0], -1)
    return result


def _normalise_weights(weights, backbones):
    if weights is None:
        return None

    weights = str(weights).strip()
    if weights == "" or weights.lower() in {"none", "null", "false", "-"}:
        return None

    upper_weights = weights.upper()
    if upper_weights in backbones.Weights.__members__:
        return backbones.Weights[upper_weights]

    parsed = urlparse(weights)
    if parsed.scheme in {"http", "https", "file"}:
        return weights

    if "/" in weights or weights.endswith(".pth"):
        path = Path(weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"DINOv3 weights file does not exist on this server: {path}. "
                "Use a shared path visible from the SLURM server node, or pass "
                "one of the built-in aliases: LVD1689M, SAT493M."
            )
        return str(path)

    return weights


def setup(
    model_name: str = "dinov3_vits16",
    weights: Optional[str] = None,
    pretrained: bool = True,
    device: Optional[int] = None,
    expected_tile_size: int = 16,
    expected_channels: int = 3,
) -> Tuple[Callable, dict]:
    """Load a DINOv3 backbone via the local hubconf.

    Parameters
    ----------
    model_name : str
        Hub entrypoint, e.g. ``dinov3_vits16``, ``dinov3_vitb16``.
    weights : str | None
        Path or URL to a checkpoint. If None and ``pretrained`` is True the
        hub default URL is used (requires network + license access).
    pretrained : bool
        Forwarded to the hub entrypoint.
    device : int | None
        CUDA device index. None → cuda:0 if available, else cpu.
    """
    torch_device = _torch_device(device)

    # Skip torch.hub.load — it spawns a subprocess for the local repo lookup
    # which can stall under nix. Import the factory directly instead.
    from dinov3.hub import backbones, classifiers, segmentors, depthers, dinotxt

    weights = _normalise_weights(weights, backbones)
    kwargs = {"pretrained": bool(pretrained)}
    if weights is not None:
        kwargs["weights"] = weights
    factories = {}
    for mod in (backbones, classifiers, segmentors, depthers, dinotxt):
        for name in getattr(mod, "__all__", []) + [
            n for n in dir(mod) if n.startswith("dinov3_")
        ]:
            fn = getattr(mod, name, None)
            if callable(fn):
                factories[name] = fn
    if model_name not in factories:
        raise ValueError(
            f"Unknown model {model_name!r}; available: {sorted(factories)[:8]}..."
        )
    loaded_model = factories[model_name](**kwargs).to(torch_device)
    loaded_model.eval()

    info = {
        "device": str(torch_device),
        "model_name": model_name,
        "repo_dir": REPO_DIR,
        "weights": weights,
        "pretrained": bool(pretrained),
        "expected_tile_size": expected_tile_size,
        "expected_channels": expected_channels,
    }

    processor = partial(
        process,
        processor=loaded_model,
        device=torch_device,
        expected_tile_size=expected_tile_size,
        expected_channels=expected_channels,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    processor: Callable,
    device,
    expected_tile_size: int,
    expected_channels: int,
) -> numpy.ndarray:
    """Run DINOv3 on an NCZYX numpy array.

    DINOv3 is a rigid 3-channel ImageNet ViT. Inputs with C ≠ 3 are split
    into ``ceil(C/3)`` 3-channel chunks via
    :func:`nahual.preprocess.channel_chunks_rigid3` (recycling leading
    channels for the trailing chunk), the backbone is run on each chunk,
    and per-chunk cls tokens are concatenated along the feature axis.
    Final shape is ``(N, D · ceil(C/3))``.
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, channels, _, *input_yx = pixels.shape
    if expected_channels is not None and channels != int(expected_channels):
        raise ValueError(
            f"Expected {expected_channels} input channels, got {channels} for shape {pixels.shape}"
        )

    from nahual.preprocess import channel_chunks_rigid3, validate_input_shape

    validate_input_shape(input_yx, expected_tile_size)

    chunks = channel_chunks_rigid3(pixels)
    outs = []
    for chunk in chunks:
        import torch

        torch_tensor = torch.from_numpy(chunk.copy()).float().to(device)
        with torch.no_grad():
            result = processor(torch_tensor)
        outs.append(_extract_numpy_features(result))
    return numpy.concatenate(outs, axis=1)


async def main(listen_address: str):
    import pynng
    from nahual.server import responder

    with pynng.Rep0(listen=listen_address, recv_timeout=300) as sock:
        print(f"DINOv3 server listening on {listen_address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    args = _parse_args()
    try:
        import trio

        trio.run(main, args.address)
    except KeyboardInterrupt:
        pass
