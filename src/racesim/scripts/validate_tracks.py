from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from racesim.env.track import ClosedTrack, TrackGeometryIssue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate configured track geometry.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/track_catalog.yaml"),
        help="Track catalog to validate when --track is not provided.",
    )
    parser.add_argument(
        "--track",
        type=Path,
        action="append",
        default=[],
        help="Specific track YAML to validate. May be provided more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.track or catalog_track_paths(args.catalog)
    results = validate_track_paths(paths)
    for path, issues in results:
        if issues:
            print(f"{path}: invalid")
            for issue in issues:
                print(f"  - {issue.message}")
        else:
            print(f"{path}: ok")

    if any(issues for _path, issues in results):
        raise SystemExit(1)


def catalog_track_paths(catalog_path: Path) -> list[Path]:
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["tracks"]
    return [Path(entry["track"]) for entry in catalog.values()]


def validate_track_paths(
    paths: list[Path],
) -> list[tuple[Path, list[TrackGeometryIssue]]]:
    results = []
    for path in paths:
        track = ClosedTrack.from_config(path)
        results.append((path, track.validate_geometry()))
    return results


if __name__ == "__main__":
    main()
