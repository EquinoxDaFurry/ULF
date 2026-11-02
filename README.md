# Unicode Lossful Format (ULF)

A deliberately silly text-based video format. ULF represents video frames using Unicode characters, choosing glyphs that visually resemble parts of the original image.

It supports audio, delta-frame compression, GPU acceleration, and MP4 export.

## Comparison

<img src="ULF_comparison.gif" alt="ULF Comparison" width="600">

## Requirements

Python 3.10+.

Install the package (with all optional playback features):

```bash
python -m pip install "unicode-lossful-format[all]"
```

FFmpeg must be installed and available on your system PATH for MP4 export.
It is an external system dependency and is not installed by pip or by any ULF
extra. Verify it with `ffmpeg -version` before using `ulf convert` or
`ulf.convert()`.

## Encode

```bash
ulf-encode input.mp4 output.ulf --size 80x40
```

### Options

- `--size WIDTHxHEIGHT`: Character-grid resolution (default: `80x40`).
- `--fps N`: Output frame rate.
- `--font PATH`: Font used for glyph matching.
- `--every N`: Encode every Nth source frame.
- `--backend auto|vulkan|cpu`: Select processing backend.
- `--keyframe-interval N`: Set the interval between full frames.
- `--mute`: Disable audio.
- `--audio-waves N`: Set audio reconstruction detail.
- `--emojiless`: Exclude emoji characters.
- `--hackerman`: Use printable keyboard ASCII characters only.
- `--debug`: Enable diagnostic output.
- `--color`: Enables colored video
- `--colour`: British Alias for --color
## Play

```bash
ulf-play output.ulf
```

### Options

- `--window WIDTHxHEIGHT`: Set playback window size.
- `--font PATH`: Select playback font.
- `--backend auto|vulkan|cpu`: Select playback backend.
- `--mute`: Disable audio.
- `--debug`: Enable diagnostic output.

## Convert to MP4

```bash
ulf convert output.ulf output.mp4 --encoder auto --export-quality high
```

The legacy equivalent remains `ulf-play output.ulf --export output.mp4`.

## Python API

The same options are available as keyword arguments, using underscores in
place of CLI hyphens:

```python
import ulf

ulf.encode("example.mp4", "example.ulf", size="80x40", backend="auto")
ulf.play("example.ulf", window="1280x720", mute=False)
ulf.convert("example.ulf", "example.mp4", encoder="auto", export_quality="high")
```

`encode()` derives an `.ulf` output path when its `output` argument is omitted.
`convert()` requires FFmpeg on your system `PATH`, just like the CLI.

## File format

ULF files use the `.ulf` extension and are stored as UTF-8 text.

The format uses the `ULF/1.0` signature and supports full-frame and delta-frame video, with optional audio data.

## Limitations

- Video is grayscale.
- Audio reconstruction is lossy.
- Visual quality depends on the selected font and renderer.
- Vulkan requires compatible hardware and drivers.
- MP4 export requires FFmpeg.
- Large resolutions can increase processing time and file size.

## Have fun

ULF makes video less efficient on purpose.

Now with compression, GPU acceleration, sound, and MP4 export!
