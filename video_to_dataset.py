"""Видео → кадры для разметки.

Нарезает видео на кадры с заданным шагом, выкидывает почти одинаковые кадры,
делит данные на train и val по видео (а не по кадрам, чтобы соседние кадры
одного ролика не оказались в обеих выборках) и раскладывает кадры по архивам
для разметчиков.

Запуск:
    python video_to_dataset.py videos out --every 1 --val 0.2 --batch 200 --labelers 4
"""
from __future__ import annotations

import argparse
import csv
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class Frame:
    video: str
    path: Path
    split: str = "train"
    batch: int = 0
    labeler: int = 0


def dhash(image, size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def extract(video: Path, out_dir: Path, every: float, near: int, start: float, end: float) -> tuple[list[Frame], int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  не открылось: {video.name}")
        return [], 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * every))
    first, last = int(start * fps), int(end * fps) if end > 0 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, taken, dropped, prev_hash, idx = [], 0, 0, None, 0
    while True:
        if last is not None and idx > last:
            break
        if idx < first or (idx - first) % step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, image = cap.read()
        if not ok:
            break
        taken += 1
        h = dhash(image)
        if prev_hash is not None and bin(h ^ prev_hash).count("1") <= near:
            dropped += 1
        else:
            path = out_dir / f"{video.stem}_{idx:06d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append(Frame(video.stem, path))
            prev_hash = h
        idx += 1
    cap.release()
    return frames, taken, dropped


def split_by_video(frames: list[Frame], val_share: float, seed: int) -> None:
    """Целые видео уходят в val, пока там не наберётся нужная доля кадров."""
    by_video: dict[str, list[Frame]] = {}
    for f in frames:
        by_video.setdefault(f.video, []).append(f)
    videos = sorted(by_video)
    random.Random(seed).shuffle(videos)
    target, in_val = val_share * len(frames), 0
    for v in videos:
        if len(videos) > 1 and in_val < target and in_val + len(by_video[v]) <= target * 1.5 + 1:
            for f in by_video[v]:
                f.split = "val"
            in_val += len(by_video[v])


def make_batches(frames: list[Frame], out: Path, batch_size: int, labelers: int) -> list[Path]:
    """Кадры одного видео держим вместе: разметчику проще, когда соседние кадры рядом."""
    ordered = sorted(frames, key=lambda f: (f.video, f.path.name))
    zips = []
    (out / "batches").mkdir(parents=True, exist_ok=True)
    for b, start in enumerate(range(0, len(ordered), batch_size), 1):
        labeler = (b - 1) % labelers + 1
        zip_path = out / "batches" / f"batch_{b:03d}_labeler_{labeler}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
            for f in ordered[start:start + batch_size]:
                f.batch, f.labeler = b, labeler
                z.write(f.path, f.path.name)
        zips.append(zip_path)
    return zips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Видео → кадры для разметки")
    ap.add_argument("videos", help="папка с видео")
    ap.add_argument("out", help="куда сложить кадры и архивы")
    ap.add_argument("--every", type=float, default=1.0, help="шаг между кадрами в секундах")
    ap.add_argument("--near", type=int, default=5, help="порог похожести на предыдущий кадр (0 — отключить)")
    ap.add_argument("--start", type=float, default=0, help="с какой секунды брать кадры")
    ap.add_argument("--end", type=float, default=0, help="до какой секунды (0 — до конца)")
    ap.add_argument("--val", type=float, default=0.2, help="доля кадров в val")
    ap.add_argument("--batch", type=int, default=200, help="кадров в одном архиве")
    ap.add_argument("--labelers", type=int, default=1, help="сколько разметчиков")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    out = Path(a.out)
    videos = sorted(p for p in Path(a.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        print(f"В «{a.videos}» нет видео ({', '.join(sorted(VIDEO_EXT))})")
        return 1
    frames, taken_total, dropped_total = [], 0, 0
    for v in videos:
        got, taken, dropped = extract(v, out / "frames" / v.stem, a.every, a.near if a.near > 0 else -1, a.start, a.end)
        print(f"{v.name}: взято кадров {taken}, оставлено {len(got)}, похожих отброшено {dropped}")
        frames += got
        taken_total += taken
        dropped_total += dropped
    if not frames:
        print("Кадров не получилось.")
        return 1

    split_by_video(frames, a.val, a.seed)
    zips = make_batches(frames, out, a.batch, a.labelers)
    for name in ("train", "val"):
        (out / f"{name}.txt").write_text("\n".join(f.path.name for f in frames if f.split == name) + "\n", encoding="utf-8")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["файл", "видео", "выборка", "архив", "разметчик"])
        for f in sorted(frames, key=lambda f: (f.batch, f.path.name)):
            w.writerow([f.path.name, f.video, f.split, f.batch, f.labeler])

    n_val = sum(f.split == "val" for f in frames)
    print(f"\nВидео: {len(videos)}. Кадров: {len(frames)} (похожих отброшено {dropped_total} из {taken_total}).")
    print(f"train: {len(frames) - n_val}, val: {n_val} — деление по видео, без пересечений.")
    print(f"Архивов для разметки: {len(zips)} в {out / 'batches'}. Сводка: {out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
