"""Unified ``ulf`` command line entry point."""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ulf", description="Unicode Lossful Format tools")
    commands = parser.add_subparsers(dest="command", required=True)
    # Disable subcommand help here so it is forwarded to the established
    # encoder/player parsers, which own the option documentation.
    commands.add_parser("encode", help="Encode video as ULF", add_help=False)
    commands.add_parser("play", help="Play a ULF file", add_help=False)
    convert_parser = commands.add_parser(
        "convert", help="Export a ULF file to MP4", add_help=False
    )
    convert_parser.add_argument("input", nargs="?")
    convert_parser.add_argument("output", nargs="?")
    namespace, remaining = parser.parse_known_args(argv)

    if namespace.command == "encode":
        from ulf_encode import main as command
        command(remaining)
    elif namespace.command == "play":
        from ulf_play import main as command
        command(remaining)
    else:
        from ulf_play import main as command
        if "-h" in remaining or "--help" in remaining:
            command(["--help"])
            return
        if namespace.input is None or namespace.output is None:
            convert_parser.error("the following arguments are required: input, output")
        command([namespace.input, "--export", namespace.output, *remaining])


if __name__ == "__main__":
    main()
