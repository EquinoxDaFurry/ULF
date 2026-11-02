"""Python API for Unicode Lossful Format (ULF).

The functions in this module expose the same settings as the command-line
tools.  Optional features need their matching extra: ``ulf[all]`` installs
GPU playback/matching and live audio playback support.
"""

from __future__ import annotations

from os import PathLike
from typing import Literal

__version__ = "2.0.0"

Pathish = str | PathLike[str]
Backend = Literal["auto", "vulkan", "cpu"]
Quality = Literal["high", "medium", "low"]
Encoder = Literal["auto", "nvenc", "amf", "qsv", "vaapi", "cpu"]


def _path(value: Pathish) -> str:
    return str(value)


def encode(
    input: Pathish,
    output: Pathish | None = None,
    *,
    size: str = "80x40",
    fps: float = 0,
    font: Pathish | None = None,
    every: int = 1,
    backend: Backend = "auto",
    keyframe_interval: int = 24,
    emojiless: bool = False,
    hackerman: bool = False,
    audio_waves: int = 8,
    audio_segment: float = 0.05,
    mute: bool = False,
    debug: bool = False,
) -> None:
    """Encode a video as ULF.

    ``output`` defaults to the input filename with a ``.ulf`` suffix.
    Keyword names correspond to the CLI options with hyphens replaced by
    underscores.
    """
    from pathlib import Path
    from ulf_encode import main

    source = _path(input)
    destination = _path(output) if output is not None else str(Path(source).with_suffix(".ulf"))
    argv = [source, destination, "--size", size, "--fps", str(fps), "--every", str(every),
            "--backend", backend, "--keyframe-interval", str(keyframe_interval),
            "--audio-waves", str(audio_waves), "--audio-segment", str(audio_segment)]
    if font is not None:
        argv += ["--font", _path(font)]
    if emojiless:
        argv.append("--emojiless")
    if hackerman:
        argv.append("--hackerman")
    if mute:
        argv.append("--mute")
    if debug:
        argv.append("--debug")
    main(argv)


def play(
    input: Pathish,
    *,
    window: str = "1000x700",
    font: Pathish | None = None,
    backend: Backend = "auto",
    mute: bool = False,
    debug: bool = False,
) -> None:
    """Play a ULF file in a window."""
    from ulf_play import main

    argv = [_path(input), "--window", window, "--backend", backend]
    if font is not None:
        argv += ["--font", _path(font)]
    if mute:
        argv.append("--mute")
    if debug:
        argv.append("--debug")
    main(argv)


def convert(
    input: Pathish,
    output: Pathish,
    *,
    font: Pathish | None = None,
    backend: Backend = "auto",
    mute: bool = False,
    export_size: str | None = None,
    export_quality: Quality = "high",
    encoder: Encoder = "auto",
) -> None:
    """Export a ULF file to MP4 using FFmpeg."""
    from ulf_play import main

    argv = [_path(input), "--backend", backend, "--export", _path(output),
            "--export-quality", export_quality, "--encoder", encoder]
    if font is not None:
        argv += ["--font", _path(font)]
    if mute:
        argv.append("--mute")
    if export_size is not None:
        argv += ["--export-size", export_size]
    main(argv)


__all__ = ["convert", "encode", "play", "__version__"]
