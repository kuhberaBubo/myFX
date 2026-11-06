"""Default checkpoint download utilities.

The StemFX checkpoint comes from HuggingFace Hub. The SCNet separator
checkpoint is fetched from the upstream Music-Source-Separation-Training
GitHub release — StemFX does not mirror those weights (see
``third_party/NOTICE``).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


DEFAULT_REPO = "barry-mir/stemfx-bsfilm"
DEFAULT_FILENAME = "best_checkpoint.pt"

# Upstream SCNet weights, downloaded on demand and cached locally.
SCNET_URL = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/"
    "download/v1.0.17/model_scnet_masked_ep_111_sdr_9.8286.ckpt"
)
SCNET_SHA256 = "b4675b0269809de27172a050e8767a857077635eda1738db0874d63a79f2b6dd"
SCNET_FILENAME = "model_scnet_masked_ep_111_sdr_9.8286.ckpt"
SCNET_CONFIG = "config_musdb18_scnet_xl_ihf.yaml"


def default_checkpoint_path(
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> str:
    """Return a local path to the default StemFX checkpoint.

    Downloads from HuggingFace Hub on first call; subsequent calls hit cache.

    Args:
        cache_dir: Override HF cache directory. Defaults to ``$HF_HOME`` /
            ``~/.cache/huggingface``.
        revision: Git revision (branch, tag, or commit SHA) on the HF repo.

    Returns:
        Absolute path to the cached checkpoint file.
    """
    repo_id = os.environ.get("STEMFX_HF_REPO", DEFAULT_REPO)
    filename = os.environ.get("STEMFX_HF_FILENAME", DEFAULT_FILENAME)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
    )


def resolve_checkpoint(
    checkpoint: Optional[str | os.PathLike] = None,
) -> str:
    """Resolve a checkpoint reference to a local file path.

    Accepts:
    - ``None``: download default checkpoint from HuggingFace Hub.
    - Local path (``str`` or ``Path``): returned as-is after existence check.
    """
    if checkpoint is None:
        return default_checkpoint_path()
    p = Path(checkpoint).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return str(p)


# --------------------------------------------------------------------------- #
#  SCNet separator assets
# --------------------------------------------------------------------------- #
def _stemfx_cache_dir() -> Path:
    """Directory for StemFX-managed downloads.

    Honours ``STEMFX_CACHE_DIR``, then ``XDG_CACHE_HOME``, falling back to
    ``~/.cache/stemfx``.
    """
    env = os.environ.get("STEMFX_CACHE_DIR")
    if env:
        base = Path(env).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "stemfx"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def default_scnet_config_path() -> str:
    """Path to the bundled SCNet YAML config (shipped inside the package)."""
    return str(Path(__file__).resolve().parent / "third_party" / "configs" / SCNET_CONFIG)


def default_scnet_path(cache_dir: Optional[str] = None, verify: bool = True) -> str:
    """Return a local path to the SCNet separator checkpoint.

    Downloads from the upstream Music-Source-Separation-Training release on
    first call (~214 MB) and caches it. StemFX does not redistribute these
    weights; they are fetched from the original project.

    Args:
        cache_dir: Override the download directory.
        verify: Check the SHA-256 of the cached/downloaded file.

    Returns:
        Absolute path to the cached checkpoint.
    """
    override = os.environ.get("STEMFX_SCNET_CKPT")
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"STEMFX_SCNET_CKPT not found: {p}")
        return str(p)

    target_dir = Path(cache_dir).expanduser() if cache_dir else _stemfx_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / SCNET_FILENAME

    if target.is_file():
        if not verify or _sha256(target) == SCNET_SHA256:
            return str(target)
        target.unlink()  # corrupt or truncated; re-fetch

    print(f"Downloading SCNet separator weights (~214 MB) from {SCNET_URL}")
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(SCNET_URL) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)
        if verify:
            got = _sha256(tmp_path)
            if got != SCNET_SHA256:
                raise RuntimeError(
                    "SCNet checkpoint failed checksum verification "
                    f"(expected {SCNET_SHA256}, got {got})."
                )
        # NamedTemporaryFile is 0600; a cache may be shared between users.
        tmp_path.chmod(0o644)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return str(target)
