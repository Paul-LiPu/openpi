#!/usr/bin/env python

"""Convert a LeRobot dataset from v3.0 layout to the legacy v2.1 layout.

This is a local filesystem converter intended for use with repos pinned to older
LeRobot versions (v2.1-era) that expect:

- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- per-episode parquet files under `data/chunk-XXX/episode_XXXXXX.parquet`
- per-episode videos under `videos/chunk-XXX/<camera>/episode_XXXXXX.mp4`

The script mirrors the structure-transforming behavior of `convert_dataset_v21_to_v30.py`
in the opposite direction. It converts in place by writing a sibling directory and then
swapping it into the original path.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd
import tqdm

V21 = "v2.1"
V30 = "v3.0"
DEFAULT_CHUNK_SIZE = 1000


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def load_parquet_tree(path: Path) -> pd.DataFrame:
    """Load parquet data from either a single file or a directory tree of parquet shards."""
    if path.is_file():
        parquet_files = [path]
    elif path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet"))
    else:
        parquet_files = []

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at/under {path}")
    frames = [pd.read_parquet(p) for p in parquet_files]
    return pd.concat(frames, ignore_index=True)


def write_jsonlines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for row in rows:
            writer.write(row)


def validate_v30_dataset(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    info = load_json(info_path)
    version = info.get("codebase_version")
    if version != V30:
        raise ValueError(f"Expected codebase_version={V30}, found {version!r} in {info_path}")
    return info


def episode_chunk_dir(ep_idx: int) -> str:
    return f"chunk-{ep_idx // DEFAULT_CHUNK_SIZE:03d}"


def episode_data_path(root: Path, ep_idx: int) -> Path:
    return root / "data" / episode_chunk_dir(ep_idx) / f"episode_{ep_idx:06d}.parquet"


def episode_video_path(root: Path, ep_idx: int, camera: str) -> Path:
    return root / "videos" / episode_chunk_dir(ep_idx) / camera / f"episode_{ep_idx:06d}.mp4"


def _infer_tasks_from_episodes(episodes_df: pd.DataFrame) -> list[dict[str, Any]]:
    if "task_index" not in episodes_df.columns:
        return []

    if "task" in episodes_df.columns:
        df = episodes_df[["task_index", "task"]].dropna().drop_duplicates().sort_values("task_index")
        return [{"task_index": int(r.task_index), "task": str(r.task)} for r in df.itertuples(index=False)]

    if "tasks" in episodes_df.columns:
        # Legacy-style episodes metadata may carry `tasks` as a list per episode. Build a stable mapping.
        task_map: dict[int, str] = {}
        for _, row in episodes_df[["task_index", "tasks"]].dropna().iterrows():
            task_idx = int(row["task_index"])
            tasks_val = row["tasks"]
            if isinstance(tasks_val, (list, tuple)) and tasks_val:
                task_map.setdefault(task_idx, str(tasks_val[0]))
            elif isinstance(tasks_val, str) and tasks_val:
                task_map.setdefault(task_idx, tasks_val)
        return [{"task_index": k, "task": v} for k, v in sorted(task_map.items())]

    return []


def _infer_tasks_from_info(info: dict[str, Any]) -> list[dict[str, Any]]:
    features = info.get("features", {})
    task_index_ft = features.get("task_index", {})
    names = task_index_ft.get("names")
    if isinstance(names, dict):
        return [{"task_index": int(k), "task": str(v)} for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if isinstance(names, list):
        return [{"task_index": i, "task": str(v)} for i, v in enumerate(names)]
    return []


def read_tasks_v30(root: Path, episodes_df: pd.DataFrame | None = None, info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    tasks_path = root / "meta" / "tasks"
    if not tasks_path.exists():
        tasks_path = root / "meta" / "tasks.parquet"
    df = load_parquet_tree(tasks_path)
    if "task_index" in df.columns and "task" in df.columns:
        df = df[["task_index", "task"]].drop_duplicates().sort_values("task_index")
        return [{"task_index": int(r.task_index), "task": str(r.task)} for r in df.itertuples(index=False)]

    if "task_index" in df.columns and episodes_df is not None:
        inferred = _infer_tasks_from_episodes(episodes_df)
        if inferred:
            return inferred

    if "task_index" in df.columns and info is not None:
        inferred = _infer_tasks_from_info(info)
        if inferred:
            return inferred

    if "task_index" in df.columns:
        # Some v3 datasets only store task indices. Synthesize stable names so v2.1 metadata is valid.
        task_ids = sorted({int(v) for v in df["task_index"].dropna().tolist()})
        logging.warning(
            "No task text found in v3 metadata; synthesizing %d task names from task_index values.", len(task_ids)
        )
        return [{"task_index": task_id, "task": f"task_{task_id}"} for task_id in task_ids]

    raise ValueError(f"Unexpected tasks schema at {tasks_path}: columns={list(df.columns)}")


def read_episodes_v30(root: Path) -> pd.DataFrame:
    episodes_path = root / "meta" / "episodes"
    if not episodes_path.exists():
        episodes_path = root / "meta" / "episodes.parquet"
    df = load_parquet_tree(episodes_path)
    if "episode_index" not in df.columns:
        raise ValueError(f"Unexpected episodes schema at {episodes_path}: missing episode_index")
    return df.sort_values("episode_index").reset_index(drop=True)


def read_episodes_stats_v30(root: Path) -> pd.DataFrame | None:
    stats_path = root / "meta" / "episodes_stats"
    if not stats_path.exists():
        stats_path = root / "meta" / "episodes_stats.parquet"
    if not stats_path.exists():
        return None
    try:
        df = load_parquet_tree(stats_path)
    except FileNotFoundError:
        return None
    if "episode_index" not in df.columns:
        return None
    return df.sort_values("episode_index").reset_index(drop=True)


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_compatible(value: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-serializable Python types."""
    value = _normalize_scalar(value)
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_compatible(value.tolist())
        except Exception:
            pass
    return value


def unflatten_stats_record(flat_record: dict[str, Any]) -> dict[str, Any]:
    """Convert {'stats/a/b': x} style keys back into nested dicts under 'stats'."""
    nested: dict[str, Any] = {}
    for key, value in flat_record.items():
        if not key.startswith("stats/"):
            continue
        parts = key.split("/")
        cur = nested
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = _json_compatible(value)
    return _json_compatible(nested.get("stats", {}))


def build_legacy_episodes_jsonl(episodes_df: pd.DataFrame, task_map: dict[int, str] | None = None) -> list[dict[str, Any]]:
    required = {"episode_index", "length"}
    missing = required - set(episodes_df.columns)
    if missing:
        raise ValueError(f"Episodes metadata missing required columns for v2.1: {sorted(missing)}")

    can_use_tasks = "tasks" in episodes_df.columns
    can_use_task_index = "task_index" in episodes_df.columns and task_map is not None
    if not can_use_tasks and not can_use_task_index:
        raise ValueError("Episodes metadata missing both `tasks` and (`task_index` + task_map`) for v2.1 conversion")

    rows = []
    for _, row in episodes_df.iterrows():
        if can_use_tasks:
            tasks_val = row["tasks"]
            tasks = list(tasks_val) if isinstance(tasks_val, (list, tuple)) else [tasks_val]
        else:
            task_idx = int(row["task_index"])
            tasks = [task_map.get(task_idx, f"task_{task_idx}")]
        rows.append(
            {
                "episode_index": int(row["episode_index"]),
                "tasks": [str(t) for t in tasks],
                "length": int(row["length"]),
            }
        )
    return rows


def build_legacy_episode_stats_jsonl(
    episodes_df: pd.DataFrame, episodes_stats_df: pd.DataFrame | None
) -> list[dict[str, Any]]:
    source_df = episodes_stats_df if episodes_stats_df is not None else episodes_df
    stat_cols = [c for c in source_df.columns if c.startswith("stats/")]
    if not stat_cols:
        raise ValueError("No flattened stats columns found in v3 metadata; cannot build episodes_stats.jsonl")

    rows = []
    for _, row in source_df[["episode_index", *stat_cols]].iterrows():
        stats = unflatten_stats_record(row.to_dict())
        rows.append({"episode_index": int(row["episode_index"]), "stats": stats})
    rows.sort(key=lambda x: x["episode_index"])
    return rows


def build_legacy_info(info_v30: dict[str, Any], data_chunks: int, total_videos: int) -> dict[str, Any]:
    info = dict(info_v30)
    info["codebase_version"] = V21
    # Restore legacy aggregate counters expected by v2.1-era loaders.
    info["total_chunks"] = int(data_chunks)
    info["total_videos"] = int(total_videos)
    # Legacy layout stores per-episode files in chunked directories.
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    if info.get("video_path") is not None:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return info


def convert_data_files(root: Path, new_root: Path, episodes_df: pd.DataFrame) -> int:
    data_cols = {"data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index", "episode_index"}
    missing = data_cols - set(episodes_df.columns)
    if missing:
        raise ValueError(f"Episodes metadata missing data columns: {sorted(missing)}")

    grouped = episodes_df.groupby(["data/chunk_index", "data/file_index"], sort=True)
    for (chunk_idx, file_idx), group in tqdm.tqdm(grouped, desc="convert data files"):
        src = root / "data" / f"chunk-{int(chunk_idx):03d}" / f"file_{int(file_idx):03d}.parquet"
        if not src.exists():
            raise FileNotFoundError(src)
        df = pd.read_parquet(src)
        group = group.sort_values("dataset_from_index")
        file_start = int(group["dataset_from_index"].min())
        for _, row in group.iterrows():
            ep_idx = int(row["episode_index"])
            local_start = int(row["dataset_from_index"]) - file_start
            local_end = int(row["dataset_to_index"]) - file_start
            ep_df = df.iloc[local_start:local_end].reset_index(drop=True)
            out = episode_data_path(new_root, ep_idx)
            out.parent.mkdir(parents=True, exist_ok=True)
            ep_df.to_parquet(out, index=False)

    return max(1, (int(episodes_df["episode_index"].max()) // DEFAULT_CHUNK_SIZE) + 1) if not episodes_df.empty else 0


def _detect_video_keys(episodes_df: pd.DataFrame) -> list[str]:
    keys: set[str] = set()
    for col in episodes_df.columns:
        if not col.startswith("videos/"):
            continue
        parts = col.split("/")
        if len(parts) >= 3:
            keys.add(parts[1])
    return sorted(keys)


def _ffmpeg_trim(src: Path, dst: Path, start_ts: float, end_ts: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end_ts) - float(start_ts))
    # First try stream copy (fast). If it fails, fall back to re-encode for robustness.
    copy_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_ts:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        str(dst),
    ]
    result = subprocess.run(copy_cmd, capture_output=True, text=True)
    if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return

    reencode_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_ts:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    subprocess.run(reencode_cmd, check=True)


def convert_videos(root: Path, new_root: Path, episodes_df: pd.DataFrame) -> int:
    video_keys = _detect_video_keys(episodes_df)
    if not video_keys:
        return 0

    for _, row in tqdm.tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="convert videos"):
        ep_idx = int(row["episode_index"])
        for camera in video_keys:
            chunk_col = f"videos/{camera}/chunk_index"
            file_col = f"videos/{camera}/file_index"
            from_col = f"videos/{camera}/from_timestamp"
            to_col = f"videos/{camera}/to_timestamp"
            if not all(c in episodes_df.columns for c in (chunk_col, file_col, from_col, to_col)):
                continue
            if pd.isna(row.get(chunk_col)):
                continue
            src = root / "videos" / camera / f"chunk-{int(row[chunk_col]):03d}" / f"file_{int(row[file_col]):03d}.mp4"
            if not src.exists():
                raise FileNotFoundError(src)
            dst = episode_video_path(new_root, ep_idx, camera)
            _ffmpeg_trim(src, dst, float(row[from_col]), float(row[to_col]))

    return len(video_keys)


def convert_dataset(repo_id: str, root: str | Path | None = None, force: bool = False) -> None:
    base = Path(root) if root is not None else Path(".")
    dataset_root = base / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    info_v30 = validate_v30_dataset(dataset_root)
    old_root = dataset_root.parent / f"{dataset_root.name}_v30_backup"
    new_root = dataset_root.parent / f"{dataset_root.name}_v21"

    if old_root.exists():
        if force:
            shutil.rmtree(old_root)
        else:
            raise FileExistsError(f"Backup directory already exists: {old_root} (use --force)")
    if new_root.exists():
        shutil.rmtree(new_root)

    logging.info("Reading v3 metadata from %s", dataset_root)
    episodes_df = read_episodes_v30(dataset_root)
    episodes_stats_df = read_episodes_stats_v30(dataset_root)
    tasks = read_tasks_v30(dataset_root, episodes_df=episodes_df, info=info_v30)

    logging.info("Converting tasks.jsonl / episodes.jsonl / episodes_stats.jsonl")
    task_map = {int(item["task_index"]): str(item["task"]) for item in tasks}
    write_jsonlines(new_root / "meta" / "tasks.jsonl", tasks)
    write_jsonlines(new_root / "meta" / "episodes.jsonl", build_legacy_episodes_jsonl(episodes_df, task_map=task_map))
    write_jsonlines(
        new_root / "meta" / "episodes_stats.jsonl",
        build_legacy_episode_stats_jsonl(episodes_df, episodes_stats_df),
    )

    stats_json = dataset_root / "meta" / "stats.json"
    if stats_json.exists():
        shutil.copy2(stats_json, new_root / "meta" / "stats.json")

    logging.info("Splitting concatenated parquet data files into per-episode files")
    data_chunks = convert_data_files(dataset_root, new_root, episodes_df)

    logging.info("Splitting concatenated video files into per-episode files")
    total_videos = convert_videos(dataset_root, new_root, episodes_df)

    info_v21 = build_legacy_info(info_v30, data_chunks=data_chunks, total_videos=total_videos)
    write_json(new_root / "meta" / "info.json", info_v21)

    logging.info("Swapping converted dataset into place")
    shutil.move(str(dataset_root), str(old_root))
    shutil.move(str(new_root), str(dataset_root))
    logging.info("Done. Backup kept at %s", old_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        required=True,
        type=str,
        help="Dataset directory name or nested path under --root (e.g. pick-place-yellow_cube).",
    )
    parser.add_argument(
        "--root",
        default=".",
        type=str,
        help="Base directory that contains the dataset folder. Dataset path = root/repo-id",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing backup directory named <dataset>_v30_backup.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    convert_dataset(repo_id=args.repo_id, root=args.root, force=args.force)


if __name__ == "__main__":
    main()
