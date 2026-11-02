import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
import wave

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ulf._optional import has_optional_dependencies, warn_optional


BINARY_AUDIO_MARKER = b"@AUDIO-BINARY-V1\n"


# ------------------------------------------------------------
# FFMPEG / ENCODER DETECTION
# ------------------------------------------------------------

ENCODER_OPTIONS = {
    "nvenc": {
        "codec": "h264_nvenc",
        "arguments": ["-preset", "p5", "-tune", "hq"],
    },
    "amf": {
        "codec": "h264_amf",
        "arguments": ["-quality", "quality"],
    },
    "qsv": {
        "codec": "h264_qsv",
        "arguments": ["-preset", "medium"],
    },
    "vaapi": {
        "codec": "h264_vaapi",
        "arguments": ["-qp", "23"],
    },
    "cpu": {
        "codec": "libx264",
        "arguments": ["-preset", "medium"],
    },
}


def get_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found in PATH. "
            "Install FFmpeg and make sure the ffmpeg command works."
        )

    return ffmpeg


def get_available_encoders(ffmpeg):
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        print("WARNING: Could not inspect FFmpeg encoders:", exc)
        return set()

    if result.returncode != 0:
        return set()

    return {
        name
        for name in (
            "h264_nvenc",
            "h264_amf",
            "h264_qsv",
            "h264_vaapi",
            "libx264",
        )
        if name in result.stdout
    }


def choose_encoder(ffmpeg, requested="auto"):
    available = get_available_encoders(ffmpeg)

    preferences = ["nvenc", "amf", "qsv", "vaapi", "cpu"]

    if requested != "auto":
        if requested not in ENCODER_OPTIONS:
            raise ValueError(f"Unknown encoder: {requested}")

        codec = ENCODER_OPTIONS[requested]["codec"]

        if codec not in available:
            raise RuntimeError(
                f"Requested encoder {codec} is not available."
            )

        return [requested]

    selected = [
        name
        for name in preferences
        if ENCODER_OPTIONS[name]["codec"] in available
    ]

    if not selected:
        raise RuntimeError(
            "FFmpeg does not expose any supported H.264 encoder."
        )

    return selected


def encoder_arguments(name, quality):
    settings = ENCODER_OPTIONS[name]

    if name == "cpu":
        crf = {"high": 18, "medium": 23, "low": 28}[quality]
        return [
            "-c:v", settings["codec"],
            *settings["arguments"],
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ]

    if name == "nvenc":
        cq = {"high": 19, "medium": 24, "low": 29}[quality]
        return [
            "-c:v", settings["codec"],
            *settings["arguments"],
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]

    if name == "amf":
        qp = {"high": 19, "medium": 24, "low": 29}[quality]
        return [
            "-c:v", settings["codec"],
            *settings["arguments"],
            "-qp_i", str(qp),
            "-qp_p", str(qp),
            "-pix_fmt", "yuv420p",
        ]

    if name == "qsv":
        global_quality = {"high": 18, "medium": 23, "low": 28}[quality]
        return [
            "-c:v", settings["codec"],
            *settings["arguments"],
            "-global_quality", str(global_quality),
            "-pix_fmt", "yuv420p",
        ]

    if name == "vaapi":
        qp = {"high": 19, "medium": 24, "low": 29}[quality]
        return [
            "-vaapi_device", "/dev/dri/renderD128",
            "-vf", "format=nv12,hwupload",
            "-c:v", settings["codec"],
            *settings["arguments"],
            "-pix_fmt", "yuv420p",
        ]

    raise ValueError(name)


# ------------------------------------------------------------
# COLOUR HELPERS
# ------------------------------------------------------------

def decode_rgb_row(line, width):
    """Decode width concatenated RRGGBB hex values."""

    expected_length = width * 6

    if len(line) != expected_length:
        raise ValueError(
            f"RGB row has {len(line)} characters; "
            f"expected {expected_length}."
        )

    try:
        return np.frombuffer(
            bytes.fromhex(line),
            dtype=np.uint8,
        ).reshape(width, 3).copy()

    except ValueError as exc:
        raise ValueError("Invalid RGB hex data.") from exc


def decode_rgb_payload(payload, count):
    """Decode a concatenated RGB hex payload."""

    if len(payload) != count * 6:
        raise ValueError(
            "Delta RGB payload length does not match its cell count."
        )

    try:
        return np.frombuffer(
            bytes.fromhex(payload),
            dtype=np.uint8,
        ).reshape(count, 3).copy()

    except ValueError as exc:
        raise ValueError("Invalid delta RGB payload.") from exc


def pack_rgb(rgb):
    """Pack an (N, 3) RGB array into uint32 values for the GPU."""

    rgb = np.asarray(rgb, dtype=np.uint32)

    return (
        rgb[:, 0]
        | (rgb[:, 1] << 8)
        | (rgb[:, 2] << 16)
    ).astype(np.uint32)


# ------------------------------------------------------------
# BINARY AUDIO
# ------------------------------------------------------------

def decode_binary_audio(payload, metadata):
    encoding = metadata.get("encoding")

    if encoding != "sine-components-binary-v1":
        raise ValueError(
            f"Unsupported binary audio encoding: {encoding!r}"
        )

    segment_count = int(metadata["segment_count"])
    hop_samples = int(metadata["hop_samples"])

    if segment_count < 0 or hop_samples < 1:
        raise ValueError("Invalid binary audio metadata.")

    segments = []
    offset = 0
    payload_length = len(payload)

    print(f"Decoding {segment_count:,} binary audio segments...")

    for segment_number in range(segment_count):
        if offset + 2 > payload_length:
            raise ValueError(
                f"Binary audio ended before segment {segment_number}."
            )

        component_count = struct.unpack_from(
            "<H", payload, offset
        )[0]

        offset += 2

        required_bytes = component_count * 6

        if offset + required_bytes > payload_length:
            raise ValueError(
                f"Binary audio segment {segment_number} is incomplete."
            )

        components = []

        for _ in range(component_count):
            frequency, amplitude, phase_q = struct.unpack_from(
                "<HeH", payload, offset
            )

            offset += 6

            phase = phase_q * (2.0 * np.pi / 65535.0)

            components.append((
                float(frequency),
                float(amplitude),
                float(phase),
            ))

        segments.append({
            "start_sample": segment_number * hop_samples,
            "components": components,
        })

        if segment_number > 0 and segment_number % 5000 == 0:
            print(
                f"\rDecoded {segment_number:,} "
                f"/ {segment_count:,} segments",
                end="",
                flush=True,
            )

    print()

    if offset != payload_length:
        print(
            f"WARNING: {payload_length - offset:,} "
            "unused bytes at the end of the audio payload."
        )

    return segments


# ------------------------------------------------------------
# ULF FILE READER
# ------------------------------------------------------------

def load_ulf(path):
    """
    Load ULF video and optional audio.

    Returns:
        header
        frames          list of character-row lists
        colours         uint8 array: (frames, height, width, 3)
        width
        height
        fps
        audio_data
    """

    raw = Path(path).read_bytes()

    binary_audio_metadata = None
    binary_audio_payload = None

    marker_position = raw.find(BINARY_AUDIO_MARKER)

    if marker_position >= 0:
        video_bytes = raw[:marker_position]

        audio_section = raw[
            marker_position + len(BINARY_AUDIO_MARKER):
        ]

        metadata_end = audio_section.find(b"\n")

        if metadata_end < 0:
            raise ValueError(
                "Binary audio metadata line is incomplete."
            )

        try:
            binary_audio_metadata = json.loads(
                audio_section[:metadata_end].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Invalid binary audio metadata."
            ) from exc

        binary_audio_payload = audio_section[metadata_end + 1:]

    else:
        video_bytes = raw

    try:
        text = video_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "ULF video section is not valid UTF-8."
        ) from exc

    import io

    file = io.StringIO(text, newline="")

    if file.readline().strip() != "ULF/1.0":
        raise ValueError("Unsupported ULF file format.")

    header = json.loads(file.readline())

    width = int(header["width"])
    height = int(header["height"])
    fps = float(header.get("fps", 25))

    if width < 1 or height < 1 or fps <= 0:
        raise ValueError("Invalid video dimensions or FPS.")

    total_cells = width * height

    frames = []
    colour_frames = []

    previous_glyphs = None
    previous_colours = None
    audio_data = None

    def read_glyph_rows():
        rows = [
            file.readline().rstrip("\r\n")
            for _ in range(height)
        ]

        if any(len(row) != width for row in rows):
            raise ValueError(
                "Invalid or incomplete full frame."
            )

        return rows

    def read_optional_keyframe_rgb():
        """
        A colour keyframe has this layout:

            @KEYFRAME n
            <glyph rows>
            @RGB
            <RGB hex rows>

        Older keyframes have no @RGB section.
        """

        position = file.tell()
        marker = file.readline()

        if marker.rstrip("\r\n") != "@RGB":
            file.seek(position)
            return np.full(
                (height, width, 3),
                255,
                dtype=np.uint8,
            )

        rgb = np.empty(
            (height, width, 3),
            dtype=np.uint8,
        )

        for y in range(height):
            line = file.readline().rstrip("\r\n")
            rgb[y] = decode_rgb_row(line, width)

        return rgb

    while True:
        marker = file.readline()

        if not marker:
            break

        marker = marker.rstrip("\r\n")

        if marker.startswith("@FRAME "):
            rows = read_glyph_rows()

            colours = np.full(
                (height, width, 3),
                255,
                dtype=np.uint8,
            )

            previous_glyphs = list("".join(rows))
            previous_colours = colours.reshape(
                total_cells, 3
            ).copy()

            frames.append(rows)
            colour_frames.append(colours)
            continue

        if marker.startswith("@KEYFRAME "):
            rows = read_glyph_rows()
            colours = read_optional_keyframe_rgb()

            previous_glyphs = list("".join(rows))
            previous_colours = colours.reshape(
                total_cells, 3
            ).copy()

            frames.append(rows)
            colour_frames.append(colours)
            continue

        if marker.startswith("@DELTA "):
            if previous_glyphs is None:
                raise ValueError(
                    "Delta frame encountered before a keyframe."
                )

            current_glyphs = previous_glyphs.copy()
            current_colours = previous_colours.copy()

            while True:
                position = file.tell()
                line = file.readline()

                if not line:
                    break

                line = line.rstrip("\r\n")

                if (
                    line.startswith("@KEYFRAME ")
                    or line.startswith("@DELTA ")
                    or line.startswith("@FRAME ")
                    or line == "@AUDIO"
                    or line == "@RGB"
                ):
                    file.seek(position)
                    break

                if not line:
                    continue

                # New colour delta format: JSON line.
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON delta record: {line[:100]!r}"
                        ) from exc

                    try:
                        start = int(record["start"])
                        count = int(record["count"])
                        glyph_payload = record["glyphs"]
                        rgb_payload = record["rgb"]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            "Delta record is missing required fields."
                        ) from exc

                    if start < 0 or count < 1:
                        raise ValueError("Invalid delta run position or count.")

                    end = start + count

                    if end > total_cells:
                        raise ValueError(
                            "Delta run extends beyond frame."
                        )

                    if len(glyph_payload) != count:
                        raise ValueError(
                            "Delta glyph payload length is invalid."
                        )

                    rgb_values = decode_rgb_payload(
                        rgb_payload,
                        count,
                    )

                    current_glyphs[start:end] = list(glyph_payload)
                    current_colours[start:end] = rgb_values

                else:
                    # Legacy monochrome delta format:
                    # start:length:glyphs
                    try:
                        start_text, length_text, payload = line.split(
                            ":", 2
                        )

                        start = int(start_text)
                        count = int(length_text)

                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid delta record: {line!r}"
                        ) from exc

                    if start < 0 or count < 1:
                        raise ValueError(
                            "Invalid delta run position or length."
                        )

                    end = start + count

                    if end > total_cells:
                        raise ValueError(
                            "Delta run extends beyond frame."
                        )

                    if len(payload) != count:
                        raise ValueError(
                            "Delta payload length does not match its record."
                        )

                    current_glyphs[start:end] = list(payload)

            previous_glyphs = current_glyphs
            previous_colours = current_colours

            rows = [
                "".join(
                    current_glyphs[y * width:(y + 1) * width]
                )
                for y in range(height)
            ]

            colours = current_colours.reshape(
                height, width, 3
            ).copy()

            frames.append(rows)
            colour_frames.append(colours)
            continue

        if marker == "@AUDIO":
            audio_data = []

            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    segment = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid audio segment: {line[:100]!r}"
                    ) from exc

                audio_data.append(segment)

            break

        raise ValueError(
            f"Invalid frame marker: {marker!r}"
        )

    if not frames:
        raise ValueError("No frames found.")

    if binary_audio_metadata is not None:
        if "audio" not in header:
            header["audio"] = binary_audio_metadata

        audio_data = decode_binary_audio(
            binary_audio_payload,
            binary_audio_metadata,
        )

        header["audio"] = {
            **header.get("audio", {}),
            **binary_audio_metadata,
        }

    if audio_data is not None and "audio" not in header:
        raise ValueError(
            "Audio section exists, but audio metadata is missing."
        )

    return (
        header,
        frames,
        np.asarray(colour_frames, dtype=np.uint8),
        width,
        height,
        fps,
        audio_data,
    )


# ------------------------------------------------------------
# AUDIO RECONSTRUCTION
# ------------------------------------------------------------

def reconstruct_audio(header, segments):
    if segments is None:
        return None, None

    metadata = header.get("audio")

    if not metadata:
        raise ValueError("Missing audio metadata.")

    sample_rate = int(metadata["sample_rate"])
    segment_samples = int(metadata["segment_samples"])
    hop_samples = int(metadata["hop_samples"])

    if (
        sample_rate < 1
        or segment_samples < 2
        or hop_samples < 1
    ):
        raise ValueError("Invalid audio metadata.")

    duration = float(metadata.get("duration", 0))

    if duration <= 0:
        raise ValueError("Invalid audio duration.")

    total_samples = int(round(duration * sample_rate))

    if total_samples <= 0:
        raise ValueError("Audio contains no samples.")

    print()
    print("Reconstructing sine-component audio...")
    print("Sample rate:", sample_rate)
    print("Audio segments:", len(segments))

    output = np.zeros(total_samples, dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)

    window = np.hanning(segment_samples).astype(np.float64)
    positions = np.arange(segment_samples, dtype=np.float64)

    angular_scale = 2.0 * np.pi / sample_rate
    total_segments = len(segments)

    for segment_number, segment in enumerate(segments):
        start = int(segment["start_sample"])

        if start < 0 or start >= total_samples:
            continue

        length = min(
            segment_samples,
            total_samples - start,
        )

        if length <= 0:
            continue

        components = segment.get("components", [])

        if components:
            component_array = np.asarray(
                components,
                dtype=np.float64,
            )

            if (
                component_array.ndim == 2
                and component_array.shape[1] == 3
            ):
                frequencies = component_array[:, 0]
                amplitudes = component_array[:, 1]
                phases = component_array[:, 2]

                valid = (
                    (frequencies > 0)
                    & (frequencies < sample_rate / 2)
                    & (amplitudes > 0)
                )

                frequencies = frequencies[valid]
                amplitudes = amplitudes[valid]
                phases = phases[valid]

                if len(frequencies):
                    angles = (
                        frequencies[:, None] * angular_scale
                        * positions[None, :length]
                        + phases[:, None]
                    )

                    reconstructed = np.sum(
                        amplitudes[:, None] * np.cos(angles),
                        axis=0,
                    )
                else:
                    reconstructed = np.zeros(
                        length,
                        dtype=np.float64,
                    )
            else:
                reconstructed = np.zeros(
                    length,
                    dtype=np.float64,
                )
        else:
            reconstructed = np.zeros(
                length,
                dtype=np.float64,
            )

        segment_window = window[:length]

        output[start:start + length] += (
            reconstructed * segment_window
        )

        weight[start:start + length] += segment_window

        if segment_number > 0 and segment_number % 2000 == 0:
            print(
                f"\rReconstructed {segment_number:,} "
                f"/ {total_segments:,} segments",
                end="",
                flush=True,
            )

    print()

    valid = weight > 1e-8
    output[valid] /= weight[valid]
    output[~valid] = 0.0

    np.clip(output, -1.0, 1.0, out=output)

    audio = output.astype(np.float32)

    print(
        f"Audio reconstructed: "
        f"{len(audio) / sample_rate:.2f} seconds"
    )

    return audio, sample_rate


# ------------------------------------------------------------
# AUDIO WAV CREATION
# ------------------------------------------------------------

def create_audio_wav(audio, sample_rate):
    if audio is None:
        return None

    if sample_rate is None or sample_rate < 1:
        raise ValueError("Invalid audio sample rate.")

    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype("<i2")

    temp_audio = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    audio_path = temp_audio.name
    temp_audio.close()

    try:
        with wave.open(audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    finally:
        del audio_int16

    return audio_path


# ------------------------------------------------------------
# MP4 EXPORT
# ------------------------------------------------------------

def export_mp4(
    output_path,
    frames,
    frame_ids,
    colours,
    tiles,
    width,
    height,
    fps,
    audio=None,
    sample_rate=None,
    export_size=None,
    quality="high",
    encoder="auto",
):
    """Render RGB frames one at a time and stream them to FFmpeg."""

    import gc

    ffmpeg = get_ffmpeg()

    output_path = Path(output_path).expanduser().resolve()

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Export output must have an .mp4 extension.")

    if not output_path.parent.exists():
        raise ValueError("The output directory does not exist.")

    frame_count = len(frame_ids)

    if frame_count == 0:
        raise ValueError("Cannot export an empty video.")

    if fps <= 0:
        raise ValueError("Invalid frame rate.")

    glyph_h = tiles.shape[1]
    glyph_w = tiles.shape[2]

    native_w = width * glyph_w
    native_h = height * glyph_h

    if export_size:
        output_w, output_h = export_size
    else:
        output_w, output_h = native_w, native_h

    if output_w < 2 or output_h < 2:
        raise ValueError("Export dimensions are too small.")

    output_w += output_w % 2
    output_h += output_h % 2

    duration = frame_count / fps

    print()
    print("=" * 60)
    print("ULF RGB MP4 EXPORT")
    print("=" * 60)

    print("Frames:", f"{frame_count:,}")
    print("FPS:", fps)
    print("Native size:", f"{native_w}x{native_h}")
    print("Output size:", f"{output_w}x{output_h}")
    print("Input pixel format: RGB24")
    print("Quality:", quality)
    print("Requested encoder:", encoder)

    audio_path = None

    if audio is not None:
        print()
        print("Preparing audio WAV...")
        audio_path = create_audio_wav(audio, sample_rate)

    encoder_candidates = choose_encoder(ffmpeg, encoder)

    print()
    print("Encoder candidates:")

    for name in encoder_candidates:
        print(
            f"  {name}: "
            f"{ENCODER_OPTIONS[name]['codec']}"
        )

    base_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{output_w}x{output_h}",
        "-framerate", str(fps),
        "-i", "pipe:0",
    ]

    if audio_path is not None:
        base_command.extend(["-i", audio_path])

    base_command.extend(["-map", "0:v:0"])

    if audio_path is not None:
        base_command.extend([
            "-map", "1:a:0",
            "-af", "apad",
            "-c:a", "aac",
            "-b:a", "192k",
        ])
    else:
        base_command.append("-an")

    selected_encoder = None
    last_error = ""

    try:
        for encoder_name in encoder_candidates:
            print()
            print(
                "Trying:",
                ENCODER_OPTIONS[encoder_name]["codec"],
            )

            command = list(base_command)

            command.extend(
                encoder_arguments(encoder_name, quality)
            )

            command.extend([
                "-t", f"{duration:.8f}",
                "-movflags", "+faststart",
                str(output_path),
            ])

            with tempfile.TemporaryFile() as error_log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=error_log,
                    bufsize=0,
                )

                try:
                    for frame_index in range(frame_count):
                        selected = tiles[frame_ids[frame_index]]

                        # Shape:
                        # (height, width, glyph_h, glyph_w)
                        mask = selected.transpose(
                            0, 2, 1, 3
                        ).reshape(
                            native_h,
                            native_w,
                        )

                        cell_colours = colours[frame_index]

                        pixel_colours = np.repeat(
                            np.repeat(
                                cell_colours,
                                glyph_h,
                                axis=0,
                            ),
                            glyph_w,
                            axis=1,
                        )

                        # Apply the cell colour to the glyph mask.
                        rgb_image = (
                            pixel_colours.astype(np.float32)
                            * (mask[:, :, None].astype(np.float32) / 255.0)
                        ).astype(np.uint8)

                        # Add a very dark background behind glyphs.
                        background = np.full_like(
                            rgb_image,
                            4,
                            dtype=np.uint8,
                        )

                        alpha = mask[:, :, None].astype(np.float32) / 255.0

                        rgb_image = (
                            rgb_image * alpha
                            + background * (1.0 - alpha)
                        ).astype(np.uint8)

                        if (
                            output_w != native_w
                            or output_h != native_h
                        ):
                            image = Image.fromarray(
                                rgb_image,
                                mode="RGB",
                            )

                            resized = image.resize(
                                (output_w, output_h),
                                Image.Resampling.NEAREST,
                            )

                            frame_bytes = resized.tobytes()

                            resized.close()
                            image.close()
                        else:
                            frame_bytes = rgb_image.tobytes()

                        process.stdin.write(frame_bytes)

                        del frame_bytes
                        del rgb_image
                        del pixel_colours
                        del mask
                        del selected

                        if (
                            frame_index % max(1, int(fps)) == 0
                            or frame_index == frame_count - 1
                        ):
                            progress = (
                                (frame_index + 1) / frame_count
                            ) * 100.0

                            print(
                                f"\rRendering: "
                                f"{frame_index + 1:,}/"
                                f"{frame_count:,} "
                                f"({progress:.1f}%)",
                                end="",
                                flush=True,
                            )

                        if frame_index > 0 and frame_index % 500 == 0:
                            gc.collect()

                    print()

                    process.stdin.close()
                    return_code = process.wait()

                except Exception:
                    if (
                        process.stdin is not None
                        and not process.stdin.closed
                    ):
                        try:
                            process.stdin.close()
                        except OSError:
                            pass

                    if process.poll() is None:
                        process.kill()

                    process.wait()

                    error_log.seek(0)
                    last_error = error_log.read().decode(
                        "utf-8",
                        errors="replace",
                    )

                    print()
                    print(
                        f"Encoder {encoder_name} failed:"
                    )
                    print(last_error[-3000:])

                    output_path.unlink(missing_ok=True)

                    if encoder != "auto":
                        raise RuntimeError(
                            f"Requested encoder {encoder_name} failed:\n"
                            f"{last_error}"
                        )

                    print("Trying the next available encoder...")
                    continue

                if return_code == 0:
                    selected_encoder = encoder_name
                    break

                error_log.seek(0)
                last_error = error_log.read().decode(
                    "utf-8",
                    errors="replace",
                )

                print()
                print(
                    f"Encoder {encoder_name} failed:"
                )
                print(last_error[-3000:])

                output_path.unlink(missing_ok=True)

                if encoder != "auto":
                    raise RuntimeError(
                        f"Requested encoder {encoder_name} failed:\n"
                        f"{last_error}"
                    )

                print("Trying the next available encoder...")

        if selected_encoder is None:
            raise RuntimeError(
                "All available encoders failed.\n" + last_error
            )

    finally:
        if audio_path is not None:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    if not output_path.is_file():
        raise RuntimeError(
            "FFmpeg finished without creating the output file."
        )

    file_size = output_path.stat().st_size

    print()
    print("=" * 60)
    print("ULF MP4 EXPORT COMPLETE")
    print("=" * 60)

    print("Output:", output_path)
    print(
        "Encoder:",
        ENCODER_OPTIONS[selected_encoder]["codec"],
    )
    print(
        "File size:",
        f"{file_size / (1024 * 1024):.2f} MiB",
    )
    print("Resolution:", f"{output_w}x{output_h}")
    print("FPS:", fps)
    print("Duration:", f"{duration:.2f} seconds")
    print("Audio:", "AAC" if audio is not None else "None")
    print()


# ------------------------------------------------------------
# AUDIO PLAYBACK
# ------------------------------------------------------------

def start_audio(audio, sample_rate, muted=False):
    if audio is None or muted:
        if muted:
            print("Audio playback: MUTED")

        return time.perf_counter()

    try:
        import sounddevice as sd

        sd.play(
            audio,
            samplerate=sample_rate,
            blocking=False,
            loop=True,
        )

        print("Audio playback: ENABLED")

    except Exception as exc:
        print()
        print("WARNING: Audio playback could not start.")
        print("Reason:", exc)
        print("Continuing with silent video playback.")
        print()

    return time.perf_counter()


# ------------------------------------------------------------
# FONT
# ------------------------------------------------------------

def find_font(explicit=None, recorded=None):
    candidates = []

    if explicit:
        candidates.append(Path(explicit))

    if recorded:
        candidates.append(Path(recorded))

    windows_fonts = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "Fonts"
    )

    for name in (
        "seguisym.ttf",
        "segoeui.ttf",
        "arial.ttf",
        "consola.ttf",
    ):
        candidates.append(windows_fonts / name)

    candidates.extend([
        Path("DejaVuSans.ttf"),
        Path("Arial.ttf"),
        Path("LiberationSans-Regular.ttf"),
    ])

    for candidate in candidates:
        try:
            if candidate.is_file():
                ImageFont.truetype(str(candidate), 20)
                return str(candidate.resolve())

            ImageFont.truetype(str(candidate), 20)
            return str(candidate)

        except OSError:
            continue

    return None


# ------------------------------------------------------------
# GLYPH ATLAS
# ------------------------------------------------------------

def build_atlas(frames, font_path, cell_w=16, cell_h=24):
    print("Finding unique characters...")

    characters = sorted({
        char
        for frame in frames
        for row in frame
        for char in row
    })

    if not characters:
        raise ValueError("No characters found.")

    if font_path:
        font = ImageFont.truetype(font_path, 20)
    else:
        font = ImageFont.load_default()

    glyph_ids = {
        char: index
        for index, char in enumerate(characters)
    }

    atlas_cols = min(
        256,
        max(1, int(np.ceil(np.sqrt(len(characters))))),
    )

    atlas_rows = int(
        np.ceil(len(characters) / atlas_cols)
    )

    atlas = Image.new(
        "RGBA",
        (atlas_cols * cell_w, atlas_rows * cell_h),
        (0, 0, 0, 255),
    )

    tiles = []

    for index, char in enumerate(characters):
        tile = Image.new(
            "L",
            (cell_w, cell_h),
            0,
        )

        draw = ImageDraw.Draw(tile)

        try:
            draw.text(
                (cell_w // 2, cell_h // 2),
                char,
                font=font,
                fill=255,
                anchor="mm",
            )
        except (ValueError, OSError):
            pass

        x = (index % atlas_cols) * cell_w
        y = (index // atlas_cols) * cell_h

        rgba = Image.merge(
            "RGBA",
            (
                tile,
                tile,
                tile,
                Image.new("L", tile.size, 255),
            ),
        )

        atlas.paste(rgba, (x, y))

        tiles.append(
            np.asarray(tile, dtype=np.uint8)
        )

    print("Converting frames to glyph IDs...")

    frame_ids = []

    for frame_number, frame in enumerate(frames):
        ids = np.empty(
            (len(frame), len(frame[0])),
            dtype=np.uint32,
        )

        for y, row in enumerate(frame):
            ids[y] = [
                glyph_ids[char]
                for char in row
            ]

        frame_ids.append(ids)

        if frame_number > 0 and frame_number % 500 == 0:
            print(
                f"\rConverted {frame_number:,} "
                f"/ {len(frames):,} frames",
                end="",
                flush=True,
            )

    print()

    return (
        characters,
        atlas,
        np.asarray(tiles),
        frame_ids,
        atlas_cols,
        cell_w,
        cell_h,
    )


# ------------------------------------------------------------
# CPU RENDERING
# ------------------------------------------------------------

def render_rgb_frame(
    frame_ids,
    colours,
    tiles,
    frame_index,
    width,
    height,
):
    """Render one frame into an RGB image."""

    selected = tiles[frame_ids[frame_index]]

    glyph_h = selected.shape[1]
    glyph_w = selected.shape[2]

    mask = selected.transpose(
        0, 2, 1, 3
    ).reshape(
        height * glyph_h,
        width * glyph_w,
    )

    cell_colours = colours[frame_index]

    pixel_colours = np.repeat(
        np.repeat(
            cell_colours,
            glyph_h,
            axis=0,
        ),
        glyph_w,
        axis=1,
    )

    alpha = mask[:, :, None].astype(np.float32) / 255.0

    background = np.full(
        pixel_colours.shape,
        4,
        dtype=np.uint8,
    )

    result = (
        pixel_colours.astype(np.float32) * alpha
        + background.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    return result


def play_cpu(
    frames,
    frame_ids,
    colours,
    tiles,
    width,
    height,
    fps,
    window_w,
    window_h,
    audio=None,
    sample_rate=None,
    muted=False,
):
    import tkinter as tk
    from PIL import ImageTk

    print()
    print("WARNING: GPU playback unavailable.")
    print("Using CPU rendering instead.")
    print()

    root = tk.Tk()

    root.title("ULF - CPU fallback")
    root.geometry(f"{window_w}x{window_h}")
    root.resizable(False, False)

    label = tk.Label(
        root,
        bd=0,
        bg="black",
    )

    label.pack(
        fill="both",
        expand=True,
    )

    delay = 1.0 / max(0.1, fps)

    playback_start = start_audio(
        audio,
        sample_rate,
        muted,
    )

    def draw_frame():
        elapsed = time.perf_counter() - playback_start
        index = int(elapsed * fps) % len(frame_ids)

        image_data = render_rgb_frame(
            frame_ids,
            colours,
            tiles,
            index,
            width,
            height,
        )

        image = Image.fromarray(
            image_data,
            mode="RGB",
        )

        image = image.resize(
            (window_w, window_h),
            Image.Resampling.NEAREST,
        )

        photo = ImageTk.PhotoImage(image)

        label.configure(image=photo)
        label.image = photo

        image.close()

        root.after(
            max(1, int(delay * 1000)),
            draw_frame,
        )

    def close():
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)

    draw_frame()
    root.mainloop()


# ------------------------------------------------------------
# VULKAN SHADER
# ------------------------------------------------------------

WGSL_SHADER = """
struct Settings {
    grid_w: u32,
    grid_h: u32,
    window_w: u32,
    window_h: u32,
    atlas_cols: u32,
    glyph_w: u32,
    glyph_h: u32,
    frame_index: u32,
};

@group(0) @binding(0)
var<uniform> settings: Settings;

@group(0) @binding(1)
var<storage, read> glyph_ids: array<u32>;

@group(0) @binding(2)
var glyph_atlas: texture_2d<f32>;

@group(0) @binding(3)
var<storage, read> cell_colours: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32
) -> VertexOutput {

    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var output: VertexOutput;

    output.position = vec4<f32>(
        positions[vertex_index],
        0.0,
        1.0
    );

    return output;
}

@fragment
fn fs_main(
    input: VertexOutput
) -> @location(0) vec4<f32> {

    let pixel = input.position.xy;

    let cell_w =
        f32(settings.window_w) / f32(settings.grid_w);

    let cell_h =
        f32(settings.window_h) / f32(settings.grid_h);

    let cell_x = min(
        u32(pixel.x / cell_w),
        settings.grid_w - 1u
    );

    let cell_y = min(
        u32(pixel.y / cell_h),
        settings.grid_h - 1u
    );

    let cell_index =
        cell_y * settings.grid_w + cell_x;

    let glyph_id = glyph_ids[cell_index];

    let local_x = min(
        u32(
            (pixel.x - f32(cell_x) * cell_w)
            / cell_w * f32(settings.glyph_w)
        ),
        settings.glyph_w - 1u
    );

    let local_y = min(
        u32(
            (pixel.y - f32(cell_y) * cell_h)
            / cell_h * f32(settings.glyph_h)
        ),
        settings.glyph_h - 1u
    );

    let atlas_x =
        (glyph_id % settings.atlas_cols)
        * settings.glyph_w + local_x;

    let atlas_y =
        (glyph_id / settings.atlas_cols)
        * settings.glyph_h + local_y;

    let mask = textureLoad(
        glyph_atlas,
        vec2<i32>(
            i32(atlas_x),
            i32(atlas_y)
        ),
        0
    ).r;

    let packed = cell_colours[cell_index];

    let red = f32(packed & 255u) / 255.0;
    let green = f32((packed >> 8u) & 255u) / 255.0;
    let blue = f32((packed >> 16u) & 255u) / 255.0;

    let foreground = vec3<f32>(
        red,
        green,
        blue
    );

    let background = vec3<f32>(
        0.015,
        0.015,
        0.015
    );

    let colour = mix(
        background,
        foreground,
        mask
    );

    return vec4<f32>(colour, 1.0);
}
"""


# ------------------------------------------------------------
# VULKAN PLAYER
# ------------------------------------------------------------

def play_vulkan(
    frames,
    frame_ids,
    colours,
    atlas,
    width,
    height,
    fps,
    window_w,
    window_h,
    atlas_cols,
    glyph_w,
    glyph_h,
    debug=False,
    audio=None,
    sample_rate=None,
    muted=False,
):
    os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"

    import wgpu
    from rendercanvas.auto import RenderCanvas, loop

    canvas = RenderCanvas(
        title="ULF - Vulkan GPU playback",
        size=(window_w, window_h),
        update_mode="continuous",
        max_fps=max(1, int(fps)),
        vsync=True,
    )

    adapter = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance",
        canvas=canvas,
    )

    if adapter is None:
        raise RuntimeError("No Vulkan adapter found.")

    print("Vulkan adapter:", adapter.info)

    device = adapter.request_device_sync()
    context = canvas.get_wgpu_context()

    render_format = context.get_preferred_format(
        device.adapter
    )

    context.configure(
        device=device,
        format=render_format,
    )

    shader = device.create_shader_module(
        code=WGSL_SHADER
    )

    pipeline = device.create_render_pipeline(
        layout=wgpu.AutoLayoutMode.auto,
        vertex={
            "module": shader,
            "entry_point": "vs_main",
        },
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{"format": render_format}],
        },
        primitive={"topology": "triangle-list"},
    )

    # --------------------------------------------------------
    # Glyph atlas.
    # --------------------------------------------------------

    atlas_data = np.asarray(
        atlas,
        dtype=np.uint8,
    )

    atlas_h, atlas_w = atlas_data.shape[:2]

    atlas_texture = device.create_texture(
        size=(atlas_w, atlas_h, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(
            wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_DST
        ),
    )

    device.queue.write_texture(
        {"texture": atlas_texture},
        atlas_data.tobytes(),
        {
            "offset": 0,
            "bytes_per_row": atlas_w * 4,
            "rows_per_image": atlas_h,
        },
        (atlas_w, atlas_h, 1),
    )

    # --------------------------------------------------------
    # Current-frame GPU buffers.
    # --------------------------------------------------------

    cells_per_frame = width * height
    buffer_bytes = cells_per_frame * 4

    glyph_buffer = device.create_buffer(
        size=max(4, buffer_bytes),
        usage=(
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
        ),
    )

    colour_buffer = device.create_buffer(
        size=max(4, buffer_bytes),
        usage=(
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
        ),
    )

    print(
        "GPU glyph buffer:",
        f"{buffer_bytes / (1024 * 1024):.2f} MiB",
    )
    print(
        "GPU colour buffer:",
        f"{buffer_bytes / (1024 * 1024):.2f} MiB",
    )

    # --------------------------------------------------------
    # Uniform buffer.
    # --------------------------------------------------------

    uniform_buffer = device.create_buffer(
        size=32,
        usage=(
            wgpu.BufferUsage.UNIFORM
            | wgpu.BufferUsage.COPY_DST
        ),
    )

    uniform_data = struct.pack(
        "<8I",
        width,
        height,
        window_w,
        window_h,
        atlas_cols,
        glyph_w,
        glyph_h,
        0,
    )

    device.queue.write_buffer(
        uniform_buffer,
        0,
        uniform_data,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": uniform_buffer},
            },
            {
                "binding": 1,
                "resource": {"buffer": glyph_buffer},
            },
            {
                "binding": 2,
                "resource": atlas_texture.create_view(),
            },
            {
                "binding": 3,
                "resource": {"buffer": colour_buffer},
            },
        ],
    )

    # --------------------------------------------------------
    # Playback.
    # --------------------------------------------------------

    playback_start = start_audio(
        audio,
        sample_rate,
        muted,
    )

    frame_count = len(frame_ids)
    previous_frame = -1

    timing_records = []
    previous_callback_time = None
    next_debug_report = 1.0
    uploaded_frame = -1

    def upload_frame(frame_index):
        nonlocal uploaded_frame

        if uploaded_frame == frame_index:
            return

        current_ids = np.ascontiguousarray(
            frame_ids[frame_index].reshape(-1),
            dtype=np.uint32,
        )

        current_colours = pack_rgb(
            colours[frame_index].reshape(-1, 3)
        )

        device.queue.write_buffer(
            glyph_buffer,
            0,
            current_ids,
        )

        device.queue.write_buffer(
            colour_buffer,
            0,
            current_colours,
        )

        uploaded_frame = frame_index

    def draw_frame():
        nonlocal previous_frame
        nonlocal previous_callback_time
        nonlocal next_debug_report

        callback_start = time.perf_counter()
        elapsed = callback_start - playback_start

        current_frame = int(elapsed * fps) % frame_count

        if previous_callback_time is None:
            callback_interval_ms = 0.0
        else:
            callback_interval_ms = (
                callback_start - previous_callback_time
            ) * 1000.0

        previous_callback_time = callback_start

        frame_advance = (
            (current_frame - previous_frame) % frame_count
            if previous_frame >= 0 else 0
        )

        if current_frame != previous_frame:
            upload_frame(current_frame)
            previous_frame = current_frame

        submit_start = time.perf_counter()

        try:
            current_texture = context.get_current_texture()
            view = current_texture.create_view()

            encoder = device.create_command_encoder()

            render_pass = encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": view,
                        "resolve_target": None,
                        "clear_value": (
                            0.015,
                            0.015,
                            0.015,
                            1.0,
                        ),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    }
                ]
            )

            render_pass.set_pipeline(pipeline)
            render_pass.set_bind_group(0, bind_group)
            render_pass.draw(3)
            render_pass.end()

            device.queue.submit([encoder.finish()])

        except wgpu.DrawCancelled:
            return

        submit_ms = (
            time.perf_counter() - submit_start
        ) * 1000.0

        callback_work_ms = (
            time.perf_counter() - callback_start
        ) * 1000.0

        if debug:
            timing_records.append((
                elapsed,
                current_frame,
                callback_interval_ms,
                submit_ms,
                callback_work_ms,
                frame_advance,
            ))

            if elapsed >= next_debug_report:
                print(
                    f"[DEBUG] "
                    f"Video: {elapsed:7.2f}s | "
                    f"Frame: {current_frame:6d} | "
                    f"Callback: {callback_interval_ms:7.2f}ms | "
                    f"Submit: {submit_ms:7.2f}ms | "
                    f"Work: {callback_work_ms:7.2f}ms | "
                    f"Advanced: {frame_advance}"
                )

                next_debug_report = elapsed + 1.0

    canvas.request_draw(draw_frame)

    print()
    print("ULF player initialized.")
    print("Playback backend: Vulkan")
    print("GPU storage: atlas + glyph buffer + colour buffer")
    print("Per-frame uploads: enabled")
    print("Debug mode:", "ON" if debug else "OFF")
    print()

    if debug:
        print("Timing data will be saved to ulf_timing.csv")

    try:
        loop.run()

    finally:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        if debug:
            with open(
                "ulf_timing.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)

                writer.writerow([
                    "time_seconds",
                    "frame_number",
                    "callback_interval_ms",
                    "cpu_submit_ms",
                    "callback_work_ms",
                    "frames_advanced",
                ])

                writer.writerows(timing_records)

            print()
            print("Performance log saved: ulf_timing.csv")
            print(
                "Recorded callbacks:",
                len(timing_records),
            )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "ULF player with RGB colour, delta decoding, Vulkan "
            "playback, binary audio, and MP4 export."
        )
    )

    parser.add_argument("input")

    parser.add_argument(
        "--window",
        default="1000x700",
    )

    parser.add_argument("--font")

    parser.add_argument(
        "--backend",
        choices=("auto", "vulkan", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Record playback timing information.",
    )

    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable audio playback.",
    )

    parser.add_argument(
        "--export",
        metavar="OUTPUT.mp4",
        help="Export the decoded ULF video directly to MP4.",
    )

    parser.add_argument(
        "--export-size",
        metavar="WIDTHxHEIGHT",
        help="MP4 output resolution.",
    )

    parser.add_argument(
        "--export-quality",
        choices=("high", "medium", "low"),
        default="high",
        help="Video quality setting.",
    )

    parser.add_argument(
        "--encoder",
        choices=(
            "auto",
            "nvenc",
            "amf",
            "qsv",
            "vaapi",
            "cpu",
        ),
        default="auto",
        help="Select the FFmpeg video encoder.",
    )

    args = parser.parse_args(argv)

    if (
        args.backend != "cpu"
        and not args.export
        and not has_optional_dependencies(
            "wgpu",
            "rendercanvas",
            "glfw",
        )
    ):
        warn_optional("GPU playback")

    if (
        not args.mute
        and not args.export
        and not has_optional_dependencies("sounddevice")
    ):
        warn_optional("live audio playback")

    # --------------------------------------------------------
    # Window dimensions.
    # --------------------------------------------------------

    try:
        window_w, window_h = map(
            int,
            args.window.lower().split("x"),
        )

        if window_w < 100 or window_h < 100:
            raise ValueError

    except ValueError:
        parser.error("--window must look like 1000x700")

    # --------------------------------------------------------
    # Export dimensions.
    # --------------------------------------------------------

    export_size = None

    if args.export_size:
        try:
            export_w, export_h = map(
                int,
                args.export_size.lower().split("x"),
            )

            if export_w < 2 or export_h < 2:
                raise ValueError

            export_size = (export_w, export_h)

        except ValueError:
            parser.error(
                "--export-size must look like 1920x1080"
            )

    # --------------------------------------------------------
    # Load ULF.
    # --------------------------------------------------------

    print("ULF player")
    print("Loading ULF file...")

    (
        header,
        frames,
        colours,
        width,
        height,
        fps,
        audio_segments,
    ) = load_ulf(args.input)

    print(f"Resolution: {width}x{height}")
    print(f"Frames: {len(frames):,}")
    print(f"FPS: {fps}")

    print(
        f"Video duration: "
        f"{len(frames) / max(fps, 0.1):.2f} seconds"
    )

    print(
        "Frame encoding:",
        header.get(
            "frame_encoding",
            "legacy full frames",
        ),
    )

    print(
        "Colour encoding:",
        header.get(
            "colour_encoding",
            "legacy monochrome / default white",
        ),
    )

    # --------------------------------------------------------
    # Audio reconstruction.
    # --------------------------------------------------------

    audio = None
    sample_rate = None

    if audio_segments is not None:
        print(
            f"Audio segments found: "
            f"{len(audio_segments):,}"
        )

        audio_metadata = header.get("audio", {})

        print(
            "Audio encoding:",
            audio_metadata.get("encoding", "legacy JSON"),
        )

        if args.mute and not args.export:
            print("Audio reconstruction skipped (--mute).")
        else:
            audio, sample_rate = reconstruct_audio(
                header,
                audio_segments,
            )
    else:
        print("No audio track found.")

    if args.mute:
        print("Audio playback: MUTED")

    # --------------------------------------------------------
    # Font.
    # --------------------------------------------------------

    font_path = find_font(
        args.font,
        header.get("font"),
    )

    if font_path:
        print("Font:", font_path)
    else:
        print("WARNING: Using default font.")

    # --------------------------------------------------------
    # Glyph atlas.
    # --------------------------------------------------------

    print("Building glyph atlas...")

    (
        characters,
        atlas,
        tiles,
        frame_ids,
        atlas_cols,
        glyph_w,
        glyph_h,
    ) = build_atlas(
        frames,
        font_path,
    )

    print(f"Unique glyphs: {len(characters):,}")

    # --------------------------------------------------------
    # Release decoded text frames.
    # --------------------------------------------------------

    frame_count = len(frames)

    del frames

    import gc
    gc.collect()

    # --------------------------------------------------------
    # MP4 export.
    # --------------------------------------------------------

    if args.export:
        del atlas
        del characters
        gc.collect()

        export_mp4(
            output_path=args.export,
            frames=range(frame_count),
            frame_ids=frame_ids,
            colours=colours,
            tiles=tiles,
            width=width,
            height=height,
            fps=fps,
            audio=audio,
            sample_rate=sample_rate,
            export_size=export_size,
            quality=args.export_quality,
            encoder=args.encoder,
        )

        return

    # --------------------------------------------------------
    # CPU playback.
    # --------------------------------------------------------

    if args.backend == "cpu":
        play_cpu(
            range(frame_count),
            frame_ids,
            colours,
            tiles,
            width,
            height,
            fps,
            window_w,
            window_h,
            audio=audio,
            sample_rate=sample_rate,
            muted=args.mute,
        )

        return

    # --------------------------------------------------------
    # Vulkan playback with automatic CPU fallback.
    # --------------------------------------------------------

    try:
        play_vulkan(
            range(frame_count),
            frame_ids,
            colours,
            atlas,
            width,
            height,
            fps,
            window_w,
            window_h,
            atlas_cols,
            glyph_w,
            glyph_h,
            debug=args.debug,
            audio=audio,
            sample_rate=sample_rate,
            muted=args.mute,
        )

    except Exception as exc:
        if args.backend == "vulkan":
            print("Vulkan playback failed:")
            print(exc)
            raise

        print()
        print("WARNING: GPU playback could not be initialized.")
        print("Reason:", exc)
        print("Falling back to CPU.")
        print()

        play_cpu(
            range(frame_count),
            frame_ids,
            colours,
            tiles,
            width,
            height,
            fps,
            window_w,
            window_h,
            audio=audio,
            sample_rate=sample_rate,
            muted=args.mute,
        )


if __name__ == "__main__":
    main()