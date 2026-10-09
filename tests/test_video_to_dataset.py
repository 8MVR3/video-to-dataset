import csv
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import video_to_dataset as V  # noqa: E402


def write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def moving(n, color):
    out = []
    for i in range(n):
        img = np.full((120, 160, 3), 30, np.uint8)
        x = 5 + i * 3
        cv2.rectangle(img, (x, 30), (x + 40, 80), color, -1)
        cv2.circle(img, (150 - i * 2, 100), 12, (255, 255, 255), -1)
        out.append(img)
    return out


def test_pipeline(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    write_video(videos / "road.avi", moving(50, (0, 0, 255)))
    write_video(videos / "yard.avi", moving(50, (0, 255, 0)))
    write_video(videos / "static.avi", [np.full((120, 160, 3), 128, np.uint8)] * 50)
    out = tmp_path / "out"
    assert V.main([str(videos), str(out), "--every", "1", "--val", "0.3", "--batch", "4", "--labelers", "2"]) == 0

    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8-sig"), delimiter=";"))
    per_video = {}
    for r in rows:
        per_video.setdefault(r["видео"], set()).add(r["выборка"])
    assert all(len(s) == 1 for s in per_video.values()), "одно видео попало в обе выборки"
    assert sum(r["видео"] == "static" for r in rows) == 1, "одинаковые кадры не отброшены"
    assert sum(r["видео"] == "road" for r in rows) == 5
    zips = sorted((out / "batches").glob("*.zip"))
    assert sum(len(zipfile.ZipFile(z).namelist()) for z in zips) == len(rows)
    assert {r["разметчик"] for r in rows} == {"1", "2"}
