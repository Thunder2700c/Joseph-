"""Validate the total size of the complete distributable AI model payload."""

from __future__ import annotations

import argparse
from pathlib import Path

from joseph.config import AI_PACKAGE_MAX_BYTES


def validate_ai_bundle(
    root: str | Path,
    *,
    max_bytes: int = AI_PACKAGE_MAX_BYTES,
) -> dict:
    """Count every regular file under root and reject packages over the hard cap.

    root must contain the complete inference payload: checkpoint, tokenizer,
    model configuration, generation metadata and any auxiliary AI assets.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if max_bytes > AI_PACKAGE_MAX_BYTES:
        raise ValueError(
            f"requested limit cannot exceed the project hard cap of "
            f"{AI_PACKAGE_MAX_BYTES} bytes"
        )

    directory = Path(root)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"AI bundle root is not a directory: {directory}")
    if directory.is_symlink():
        raise ValueError("AI bundle root cannot be a symbolic link")

    files: list[dict] = []
    total_bytes = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"AI bundle must not contain symbolic links: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({
            "path": str(path.relative_to(directory)),
            "bytes": size,
        })
        if total_bytes > max_bytes:
            raise ValueError(
                f"AI bundle is {total_bytes} bytes, above the hard limit "
                f"of {max_bytes} bytes"
            )

    return {
        "root": str(directory),
        "total_bytes": total_bytes,
        "max_bytes": max_bytes,
        "remaining_bytes": max_bytes - total_bytes,
        "file_count": len(files),
        "files": files,
        "within_limit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that the complete inference AI payload is at most 1,500,000,000 bytes."
    )
    parser.add_argument("--root", required=True, help="directory containing every distributable AI asset")
    args = parser.parse_args()

    report = validate_ai_bundle(args.root)
    print(
        f"AI bundle size: {report['total_bytes']:,} / "
        f"{report['max_bytes']:,} bytes "
        f"({report['remaining_bytes']:,} bytes remaining)"
    )
    print(f"Files counted: {report['file_count']}")


if __name__ == "__main__":
    main()
