
#!/usr/bin/env python3
"""ULF encoder with Vulkan glyph matching, delta frames, and compact binary audio."""

import argparse
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont


# ------------------------------------------------------------
# EMOJI FILTER
# ------------------------------------------------------------

def is_emoji(ch):
    """Return True for emoji and emoji-presentation characters."""

    try:
        import regex

        return bool(
            regex.match(
                r"\p{Extended_Pictographic}|\p{Emoji_Presentation}",
                ch,
            )
        )

    except ImportError:
        cp = ord(ch)

        return (
            0x1F000 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0x2300 <= cp <= 0x23FF
            or 0x2B00 <= cp <= 0x2BFF
            or cp == 0xFE0F
            or 0x1F3FB <= cp <= 0x1F3FF
        )


# ------------------------------------------------------------
# FONT AND GLYPH TEMPLATES
# ------------------------------------------------------------

def find_font(path=None):
    if path:
        p = Path(path).expanduser()

        if not p.is_file():
            raise FileNotFoundError(f"Font not found: {p}")

        return str(p.resolve())

    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    font_dir = windir / "Fonts"

    candidates = [
        font_dir / name
        for name in (
            "seguisym.ttf",
            "seguiemj.ttf",
            "segoeui.ttf",
            "arial.ttf",
            "arialuni.ttf",
            "consola.ttf",
        )
    ]

    candidates.extend([
        Path("DejaVuSans.ttf"),
        Path("Arial.ttf"),
        Path("LiberationSans-Regular.ttf"),
    ])

    for candidate in candidates:
        try:
            if candidate.is_file():
                ImageFont.truetype(str(candidate), 18)
                return str(candidate.resolve())

            ImageFont.truetype(str(candidate), 18)
            return str(candidate)

        except OSError:
            continue

    raise RuntimeError(
        "Could not find a default font. Specify --font."
    )


def build_glyph_templates(
    font_path,
    cell_w=12,
    cell_h=20,
    emojiless=False,
    hackerman=False,
):
    ft = TTFont(font_path, fontNumber=0)
    cmap = ft.getBestCmap() or {}
    font = ImageFont.truetype(font_path, 18)

    templates = []
    chars = []

    for cp in sorted(cmap):
        if cp < 32 or 0xD800 <= cp <= 0xDFFF:
            continue

        if cp > 0x10FFFF:
            continue

        ch = chr(cp)

        if hackerman and not (33 <= cp <= 126):
            continue

        if emojiless and is_emoji(ch):
            continue

        category = unicodedata.category(ch)

        if category[0] in ("C", "M", "Z") or ch.isspace():
            continue

        canvas = Image.new("L", (cell_w, cell_h), 0)
        draw = ImageDraw.Draw(canvas)

        try:
            draw.text(
                (cell_w / 2, cell_h / 2),
                ch,
                font=font,
                fill=255,
                anchor="mm",
            )

        except (ValueError, OSError):
            continue

        if canvas.getbbox() is None:
            continue

        tiny = canvas.resize((8, 8), Image.Resampling.BOX)

        templates.append(
            np.asarray(tiny, dtype=np.float32).reshape(-1) / 255.0
        )

        chars.append(ch)

    if not templates:
        raise RuntimeError(
            "No printable glyphs found in this font."
        )

    return chars, np.asarray(templates, dtype=np.float32)


# ------------------------------------------------------------
# CPU GLYPH MATCHING
# ------------------------------------------------------------

def nearest_glyph_ids(tile_data, templates, means):
    """CPU fallback. Returns one glyph ID per 8x8 tile."""

    result = np.empty(len(tile_data), dtype=np.uint32)

    for i, tile in enumerate(tile_data):
        target = tile.astype(np.float32) / 255.0
        target = target.reshape(-1)

        mean = float(target.mean())
        pos = int(np.searchsorted(means, mean))

        lo = max(0, pos - 64)
        hi = min(len(means), pos + 64)

        candidates = templates[lo:hi]

        errors = np.mean(
            (candidates - target[None, :]) ** 2,
            axis=1,
        )

        result[i] = lo + int(np.argmin(errors))

    return result


# ------------------------------------------------------------
# VULKAN GPU GLYPH MATCHER
# ------------------------------------------------------------

GPU_SHADER = """
struct Params {
    cell_count: u32,
    glyph_count: u32,
    padding0: u32,
    padding1: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> tiles: array<u32>;

@group(0) @binding(2)
var<storage, read> templates: array<f32>;

@group(0) @binding(3)
var<storage, read> means: array<f32>;

@group(0) @binding(4)
var<storage, read_write> output_ids: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;

    if (cell >= params.cell_count) {
        return;
    }

    let base = cell * 64u;

    var target_mean = 0.0;

    for (var i = 0u; i < 64u; i++) {
        target_mean += f32(tiles[base + i]) / 255.0;
    }

    target_mean /= 64.0;

    var left = 0u;
    var right = params.glyph_count;

    while (left < right) {
        let middle = left + (right - left) / 2u;

        if (means[middle] < target_mean) {
            left = middle + 1u;
        } else {
            right = middle;
        }
    }

    var lo = 0u;

    if (left > 64u) {
        lo = left - 64u;
    }

    let hi = min(params.glyph_count, left + 64u);

    var best_id = lo;
    var best_error = 1.0e30;

    for (var glyph = lo; glyph < hi; glyph++) {
        var error = 0.0;

        for (var p = 0u; p < 64u; p++) {
            let target_value = f32(tiles[base + p]) / 255.0;
            let difference = templates[glyph * 64u + p] - target_value;
            error += difference * difference;
        }

        error /= 64.0;

        if (error < best_error) {
            best_error = error;
            best_id = glyph;
        }
    }

    output_ids[cell] = best_id;
}
"""


class VulkanGlyphMatcher:
    """Matches all cells in one frame using a Vulkan compute shader."""

    def __init__(self, templates, means):
        os.environ["WGPU_BACKEND_TYPE"] = "Vulkan"

        import wgpu

        self.wgpu = wgpu

        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )

        if adapter is None:
            raise RuntimeError(
                "No Vulkan-compatible GPU adapter found."
            )

        print("Encoder Vulkan adapter:", adapter.info)

        self.device = adapter.request_device_sync()
        self.glyph_count = len(means)

        shader = self.device.create_shader_module(code=GPU_SHADER)

        self.pipeline = self.device.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
            },
        )

        template_data = np.ascontiguousarray(
            templates.reshape(-1),
            dtype=np.float32,
        )

        mean_data = np.ascontiguousarray(
            means,
            dtype=np.float32,
        )

        self.template_buffer = self.device.create_buffer(
            size=template_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.means_buffer = self.device.create_buffer(
            size=mean_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.device.queue.write_buffer(
            self.template_buffer, 0, template_data
        )

        self.device.queue.write_buffer(
            self.means_buffer, 0, mean_data
        )

        self.params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.input_buffer = None
        self.output_buffer = None
        self.readback_buffer = None
        self.bind_group = None
        self.cell_count = None

        print(
            f"Uploaded {self.glyph_count:,} glyph templates "
            "to GPU memory."
        )

    def _prepare_frame_buffers(self, cell_count):
        wgpu = self.wgpu

        if self.cell_count == cell_count:
            return

        self.cell_count = cell_count

        input_bytes = cell_count * 64 * 4
        output_bytes = cell_count * 4

        self.input_buffer = self.device.create_buffer(
            size=input_bytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.output_buffer = self.device.create_buffer(
            size=output_bytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        self.readback_buffer = self.device.create_buffer(
            size=output_bytes,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        self.bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self.params_buffer},
                },
                {
                    "binding": 1,
                    "resource": {"buffer": self.input_buffer},
                },
                {
                    "binding": 2,
                    "resource": {"buffer": self.template_buffer},
                },
                {
                    "binding": 3,
                    "resource": {"buffer": self.means_buffer},
                },
                {
                    "binding": 4,
                    "resource": {"buffer": self.output_buffer},
                },
            ],
        )

    def match(self, tile_data):
        wgpu = self.wgpu

        tile_data = np.ascontiguousarray(
            tile_data.reshape(-1),
            dtype=np.uint32,
        )

        cell_count = len(tile_data) // 64
        self._prepare_frame_buffers(cell_count)

        self.device.queue.write_buffer(
            self.input_buffer,
            0,
            tile_data,
        )

        self.device.queue.write_buffer(
            self.params_buffer,
            0,
            np.asarray(
                [cell_count, self.glyph_count, 0, 0],
                dtype=np.uint32,
            ),
        )

        encoder = self.device.create_command_encoder()

        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self.bind_group)
        compute_pass.dispatch_workgroups((cell_count + 63) // 64)
        compute_pass.end()

        encoder.copy_buffer_to_buffer(
            self.output_buffer,
            0,
            self.readback_buffer,
            0,
            cell_count * 4,
        )

        self.device.queue.submit([encoder.finish()])

        self.readback_buffer.map_sync(wgpu.MapMode.READ)

        try:
            result = np.frombuffer(
                self.readback_buffer.read_mapped(),
                dtype=np.uint32,
                count=cell_count,
            ).copy()

        finally:
            self.readback_buffer.unmap()

        return result


# ------------------------------------------------------------
# DELTA FRAME ENCODING
# ------------------------------------------------------------

def make_delta_records(previous, current, max_gap=2):
    """Create changed runs. Small unchanged gaps are included in runs."""

    changed = np.flatnonzero(previous != current)

    if len(changed) == 0:
        return []

    groups = []
    start = int(changed[0])
    end = start

    for value in changed[1:]:
        value = int(value)

        if value - end - 1 <= max_gap:
            end = value
        else:
            groups.append((start, end))
            start = end = value

    groups.append((start, end))

    records = []

    for start, end in groups:
        payload = "".join(current[start:end + 1].tolist())
        records.append(
            f"{start}:{end - start + 1}:{payload}\n"
        )

    return records


def write_keyframe(out, frame_number, rows):
    out.write(f"@KEYFRAME {frame_number}\n")

    for row in rows:
        out.write(row + "\n")


def write_delta(out, frame_number, records):
    out.write(f"@DELTA {frame_number}\n")

    for record in records:
        out.write(record)


# ------------------------------------------------------------
# AUDIO EXTRACTION
# ------------------------------------------------------------

def extract_audio(input_path, sample_rate=22050):
    """
    Extract mono float32 audio using FFmpeg.

    Returns a NumPy array containing the decoded audio samples.
    """

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError(
            "Audio is enabled by default, but FFmpeg was not found. "
            "Install FFmpeg and add it to PATH, or use --mute."
        )

    with tempfile.NamedTemporaryFile(
        suffix=".f32",
        delete=False,
    ) as temp:
        temp_path = temp.name

    try:
        command = [
            ffmpeg,
            "-v", "error",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            temp_path,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg audio extraction failed:\n"
                + result.stderr.strip()
            )

        audio = np.fromfile(temp_path, dtype="<f4")

        if len(audio) == 0:
            print("Input has no audio track. Continuing without audio.")
            return None

        return audio

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ------------------------------------------------------------
# COMPACT BINARY AUDIO ANALYSIS
# ------------------------------------------------------------

def analyse_audio(
    audio,
    sample_rate,
    wave_count,
    segment_seconds,
):
    """
    Analyse overlapping Hann-windowed audio segments.

    Binary component layout:
        uint16 frequency_hz
        float16 amplitude
        uint16 phase_quantized

    Each segment begins with:
        uint16 component_count

    All values are little-endian.

    Frequency is rounded to the nearest whole Hz.
    Phase maps [0, 2*pi) to [0, 65535].
    Amplitude is stored as IEEE-754 float16.
    """

    segment_samples = max(
        256,
        int(round(segment_seconds * sample_rate)),
    )

    hop_samples = max(1, segment_samples // 2)

    window = np.hanning(segment_samples).astype(np.float32)
    window_sum = float(window.sum())

    if window_sum <= 0:
        raise RuntimeError("Invalid audio analysis window.")

    total_samples = len(audio)
    payload = bytearray()

    segment_number = 0
    total_components = 0

    frequencies = np.fft.rfftfreq(
        segment_samples,
        d=1.0 / sample_rate,
    )

    nyquist = sample_rate // 2

    for start in range(0, total_samples, hop_samples):
        chunk = audio[start:start + segment_samples]

        if len(chunk) == 0:
            break

        padded = np.zeros(segment_samples, dtype=np.float32)
        padded[:len(chunk)] = chunk

        windowed = padded * window

        spectrum = np.fft.rfft(windowed)
        magnitudes = np.abs(spectrum)

        # Exclude DC.
        if len(magnitudes) > 0:
            magnitudes[0] = 0.0

        # Ignore the Nyquist component.
        if len(magnitudes) > 1:
            magnitudes[-1] = 0.0

        available = max(0, len(magnitudes) - 1)
        count = min(wave_count, available)

        components = []

        if count:
            candidate_indices = np.argpartition(
                magnitudes,
                -count,
            )[-count:]

            candidate_indices = sorted(
                candidate_indices,
                key=lambda index: magnitudes[index],
                reverse=True,
            )

            for index in candidate_indices:
                amplitude = (
                    2.0 * float(magnitudes[index]) / window_sum
                )

                phase = float(np.angle(spectrum[index]))
                frequency = float(frequencies[index])

                if amplitude <= 1e-8:
                    continue

                # Frequency: unsigned 16-bit integer in Hz.
                frequency_q = max(
                    0,
                    min(nyquist, int(round(frequency))),
                )

                # Amplitude: IEEE-754 half-precision float.
                amplitude_q = np.float16(amplitude)

                # Phase: unsigned 16-bit turn around the circle.
                phase_wrapped = phase % (2.0 * np.pi)

                phase_q = int(round(
                    phase_wrapped * 65535.0 / (2.0 * np.pi)
                )) & 0xFFFF

                components.append((
                    frequency_q,
                    amplitude_q,
                    phase_q,
                ))

        # A segment can never contain more than 65535 components.
        if len(components) > 65535:
            raise RuntimeError(
                "Too many audio components in one segment."
            )

        # Write segment component count.
        payload.extend(struct.pack("<H", len(components)))

        # Write each compact component.
        for frequency_q, amplitude_q, phase_q in components:
            payload.extend(
                struct.pack(
                    "<HeH",
                    frequency_q,
                    float(amplitude_q),
                    phase_q,
                )
            )

        total_components += len(components)
        segment_number += 1

        if segment_number % 1000 == 0:
            print(
                f"\rAnalysed {segment_number:,} audio segments",
                end="",
                flush=True,
            )

    print()

    metadata = {
        "sample_rate": sample_rate,
        "channels": 1,
        "encoding": "sine-components-binary-v1",
        "window": "hann",
        "segment_samples": segment_samples,
        "hop_samples": hop_samples,
        "segment_seconds": segment_samples / sample_rate,
        "duration": total_samples / sample_rate,
        "wave_count": wave_count,
        "segment_count": segment_number,
        "component_count": total_components,
        "component_layout": "uint16-float16-uint16",
        "byte_order": "little",
    }

    return {
        "metadata": metadata,
        "payload": bytes(payload),
        "segment_count": segment_number,
        "component_count": total_components,
    }


def write_audio(out_path, audio_data):
    """
    Append the compact binary audio section.

    File layout:
        @AUDIO-BINARY-V1\\n
        JSON metadata\\n
        binary segment payload

    The binary payload ends at EOF.
    """

    if audio_data is None:
        return

    metadata = audio_data["metadata"]

    with out_path.open("ab") as out:
        out.write(b"@AUDIO-BINARY-V1\n")

        metadata_line = json.dumps(
            metadata,
            separators=(",", ":"),
        ).encode("utf-8")

        out.write(metadata_line)
        out.write(b"\n")

        out.write(audio_data["payload"])


# ------------------------------------------------------------
# MAIN ENCODER
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Encode video as Unicode Lossful Format "
            "with delta frames and compact binary audio."
        )
    )

    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--size", default="80x40")
    parser.add_argument("--fps", type=float, default=0)
    parser.add_argument("--font")
    parser.add_argument("--every", type=int, default=1)

    parser.add_argument(
        "--backend",
        choices=("auto", "vulkan", "cpu"),
        default="auto",
        help="Glyph matching backend.",
    )

    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=24,
        help="Write a full keyframe every N encoded frames.",
    )

    parser.add_argument(
        "--emojiless",
        action="store_true",
        help="Exclude emoji characters from the Unicode glyph library.",
    )

    parser.add_argument(
        "--hackerman",
        action="store_true",
        help="Use only printable US keyboard ASCII characters.",
    )

    parser.add_argument(
        "--audio-waves",
        type=int,
        default=8,
        help="Number of sine-wave components per audio segment.",
    )

    parser.add_argument(
        "--audio-segment",
        type=float,
        default=0.05,
        help="Audio analysis window duration in seconds.",
    )

    parser.add_argument(
        "--mute",
        action="store_true",
        help="Do not extract or include audio.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-frame encoding timings.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate settings.
    # --------------------------------------------------------

    try:
        width, height = (
            int(value)
            for value in args.size.lower().split("x", 1)
        )

        if width < 1 or height < 1:
            raise ValueError

    except ValueError:
        parser.error("--size must look like 80x40")

    if args.every < 1:
        parser.error("--every must be at least 1")

    if args.keyframe_interval < 1:
        parser.error("--keyframe-interval must be at least 1")

    if args.audio_waves < 1 or args.audio_waves > 65535:
        parser.error("--audio-waves must be between 1 and 65535")

    if args.audio_segment <= 0:
        parser.error("--audio-segment must be greater than zero")

    if args.audio_segment > 1.0:
        parser.error("--audio-segment must not exceed 1 second")

    font_path = find_font(args.font)

    print("ULF encoder")
    print("Font:", font_path)

    if args.emojiless:
        print("Emoji filtering: ENABLED")

    if args.hackerman:
        print("HACKERMAN MODE: ENABLED")
        print("Glyph restriction: printable US keyboard ASCII")

    if args.mute:
        print("Audio: MUTED")
    else:
        print("Audio: ENABLED")
        print("Audio waves per segment:", args.audio_waves)
        print("Audio segment duration:", args.audio_segment, "seconds")
        print("Audio storage: COMPACT BINARY")

    # --------------------------------------------------------
    # Build glyph library.
    # --------------------------------------------------------

    print("Building glyph library...")

    chars, templates = build_glyph_templates(
        font_path,
        emojiless=args.emojiless,
        hackerman=args.hackerman,
    )

    means = templates.mean(axis=1)

    order = np.argsort(means)
    means = means[order]
    templates = templates[order]
    chars = [chars[i] for i in order]

    print(f"Loaded {len(chars):,} printable glyphs.")

    # --------------------------------------------------------
    # Initialize matching backend.
    # --------------------------------------------------------

    matcher = None
    backend = "CPU"

    if args.backend != "cpu":
        try:
            matcher = VulkanGlyphMatcher(templates, means)
            backend = "Vulkan GPU"

        except Exception as exc:
            if args.backend == "vulkan":
                raise RuntimeError(
                    f"Vulkan encoder initialization failed: {exc}"
                ) from exc

            print()
            print("WARNING: Vulkan encoder unavailable.")
            print("Reason:", exc)
            print("Falling back to CPU glyph matching.")
            print()

    print("Glyph matching backend:", backend)

    # --------------------------------------------------------
    # Open video.
    # --------------------------------------------------------

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = args.fps if args.fps > 0 else source_fps / args.every

    source_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )

    # --------------------------------------------------------
    # Extract and analyse audio.
    # --------------------------------------------------------

    audio_data = None

    if not args.mute:
        print()
        print("Extracting audio...")

        audio_samples = extract_audio(args.input)

        if audio_samples is not None:
            print(
                f"Extracted {len(audio_samples):,} audio samples."
            )

            print("Analysing audio into compact binary components...")

            audio_data = analyse_audio(
                audio_samples,
                sample_rate=22050,
                wave_count=args.audio_waves,
                segment_seconds=args.audio_segment,
            )

            print(
                f"Created {audio_data['segment_count']:,} "
                "audio segments."
            )

            print(
                f"Audio duration: "
                f"{audio_data['metadata']['duration']:.2f} seconds"
            )

            print(
                f"Audio components: "
                f"{audio_data['component_count']:,}"
            )

            print(
                f"Binary audio payload: "
                f"{len(audio_data['payload']):,} bytes "
                f"({len(audio_data['payload']) / 1024:.1f} KiB)"
            )

        del audio_samples

    # --------------------------------------------------------
    # ULF header.
    # --------------------------------------------------------

    out_path = Path(args.output)

    header = {
        "width": width,
        "height": height,
        "fps": fps,
        "source": Path(args.input).name,
        "glyph_count": len(chars),
        "font": font_path,
        "colour": False,
        "frame_encoding": "delta-rle-v1",
        "keyframe_interval": args.keyframe_interval,
        "audio_encoding": (
            "sine-components-binary-v1"
            if audio_data is not None
            else None
        ),
    }

    if audio_data is not None:
        header["audio"] = audio_data["metadata"]

    # --------------------------------------------------------
    # Encode video frames.
    # --------------------------------------------------------

    encoded = 0
    source_index = 0
    previous_ids = None

    total_full_bytes = 0
    total_delta_bytes = 0

    started = time.perf_counter()

    try:
        with out_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as out:

            out.write("ULF/1.0\n")
            out.write(
                json.dumps(header, ensure_ascii=False) + "\n"
            )

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                current_source_index = source_index
                source_index += 1

                if current_source_index % args.every:
                    continue

                frame_start = time.perf_counter()

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                resized = cv2.resize(
                    gray,
                    (width * 8, height * 8),
                    interpolation=cv2.INTER_AREA,
                )

                tile_data = (
                    resized
                    .reshape(height, 8, width, 8)
                    .transpose(0, 2, 1, 3)
                    .reshape(width * height, 64)
                )

                if matcher is not None:
                    ids = matcher.match(tile_data)

                else:
                    ids = nearest_glyph_ids(
                        tile_data,
                        templates,
                        means,
                    )

                ids = np.asarray(ids, dtype=np.uint32)

                current_flat = np.asarray(
                    chars,
                    dtype=object,
                )[ids]

                full_rows = [
                    "".join(
                        current_flat[
                            y * width:(y + 1) * width
                        ].tolist()
                    )
                    for y in range(height)
                ]

                force_keyframe = (
                    previous_ids is None
                    or encoded % args.keyframe_interval == 0
                )

                if force_keyframe:
                    write_keyframe(
                        out,
                        encoded,
                        full_rows,
                    )

                    total_full_bytes += sum(
                        len(row.encode("utf-8")) + 1
                        for row in full_rows
                    )

                else:
                    records = make_delta_records(
                        previous_ids,
                        current_flat,
                    )

                    delta_size = (
                        len(
                            f"@DELTA {encoded}\n".encode("utf-8")
                        )
                        + sum(
                            len(record.encode("utf-8"))
                            for record in records
                        )
                    )

                    full_size = (
                        len(
                            f"@KEYFRAME {encoded}\n".encode("utf-8")
                        )
                        + sum(
                            len(row.encode("utf-8")) + 1
                            for row in full_rows
                        )
                    )

                    if delta_size >= full_size * 0.85:
                        write_keyframe(
                            out,
                            encoded,
                            full_rows,
                        )

                        total_full_bytes += full_size

                    else:
                        write_delta(
                            out,
                            encoded,
                            records,
                        )

                        total_delta_bytes += delta_size

                previous_ids = current_flat.copy()
                encoded += 1

                if encoded % 10 == 0:
                    suffix = (
                        f"/{source_frame_count}"
                        if source_frame_count else ""
                    )

                    print(
                        f"\rEncoded {encoded}{suffix} frames",
                        end="",
                        flush=True,
                    )

                if args.debug:
                    frame_ms = (
                        time.perf_counter() - frame_start
                    ) * 1000

                    print(
                        f"\n[DEBUG] Frame {encoded - 1}: "
                        f"{frame_ms:.2f} ms"
                    )

    finally:
        cap.release()

    # --------------------------------------------------------
    # Append binary audio after the text video section.
    # --------------------------------------------------------

    if audio_data is not None:
        print()
        print("Writing compact binary audio...")
        write_audio(out_path, audio_data)
        print("Binary audio section written.")

    # --------------------------------------------------------
    # Final statistics.
    # --------------------------------------------------------

    elapsed = time.perf_counter() - started

    output_size = (
        out_path.stat().st_size
        if out_path.exists()
        else 0
    )

    print()
    print(f"Wrote {encoded:,} frames to {out_path}")
    print(f"Encoding backend: {backend}")
    print(f"Encoding time: {elapsed:.2f} seconds")
    print(f"Output size: {output_size / (1024 * 1024):.2f} MiB")
    print(f"Keyframe character data: {total_full_bytes:,} bytes")
    print(f"Delta data: {total_delta_bytes:,} bytes")

    if audio_data is not None:
        print(
            "Audio segments:",
            f"{audio_data['segment_count']:,}",
        )
        print(
            "Audio components:",
            f"{audio_data['component_count']:,}",
        )
        print(
            "Audio payload:",
            f"{len(audio_data['payload']):,} bytes",
        )
        print("Audio format: sine-components-binary-v1")
    else:
        print("Audio: none")


if __name__ == "__main__":
    main()
