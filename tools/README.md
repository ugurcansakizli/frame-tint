# Tools

Helper scripts used while developing Frame Tint. They run on the headset, use the installed Frame Tint driver
(start Frame Tint once so the camera driver is loaded; stop it first for tools that start their own `ftoverlay`),
and read `ftgrab`/`ftoverlay` from `~/.local/share/frametint/bin` (override with `FT_HOME`).
Run them with the install's Python: `~/.local/share/frametint/venv/bin/python tools/<name>.py`.

| Script | What it does |
|---|---|
| `cam_check.py` | Watches both front cameras for 8 s and prints frame rate, gaps, brightness and pairing quality. |
| `record.py` | Records 30 s of right-camera pictures plus head tracking into `frametint-rec.npz` (needs `ftgrab 0 0 . 1 \| python tools/record.py`). |
| `record_pair.py` | Records about 9 s of both cameras, matched frame by frame, plus head tracking, into `frametint-pair.npz`. |
| `stereo_test.py` | Black-and-white test of one view per eye (no colors), with a pairing and timing readout. |
| `gaze_probe.cpp` | Checks that a background SteamVR app can read where the eyes look (C++; build command at the top of the file). Prints gaze angles for 20 s. |
| `ab_snap.py` | Saves preview pictures of the running Frame Tint with different live settings (`/set`), for side-by-side comparison. |
| `flicker.py` | Measures how much the colors flicker (head kept still) for different live settings. |
| `bench.py` | Times the color model for different thread counts and core sets: `python tools/bench.py model.onnx`. |

Recordings (`*.npz`) contain pictures of your room: they are in `.gitignore`, do not publish them.
