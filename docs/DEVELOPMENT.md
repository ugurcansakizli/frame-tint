# Development notes

## Build and run from a checkout

On the headset:

```
bash scripts/build.sh                  # build/driver_frametint.so, build/ftgrab, build/ftoverlay
bash install.sh                        # installs the build, venv, driver registration, launcher
```

After changing only Python (`src/server/*.py`), copy the file to `~/.local/share/frametint/lib/` and restart Frame Tint.
After changing `ftoverlay` or `ftgrab`: `bash scripts/build.sh && install -m 755 build/ftoverlay build/ftgrab ~/.local/share/frametint/bin/`.
After changing the **driver**: reinstall and restart SteamVR (it is loaded into `vrserver`).

Run in a terminal to see everything printed (stop the launcher version first with `~/.local/bin/frametint-stop`):

```
cd ~/.local/share/frametint/lib && set -a && . ~/.config/frametint/frametint.conf && set +a && ../venv/bin/python live_server.py
```

`live_server.py` also takes `--stdin` (read right-camera frames from a pipe, for testing with a fake or recorded source), and
`FT_OVERLAY=0` runs without the in-headset panel (status page only). `FT_GRAB`, `FT_GRAB_LEFT` and `FT_OVERLAY_CMD` replace the
helper commands, which is how the author tested off the headset with fake grabbers feeding recorded frames and a fake
`ftoverlay` that speaks the pipe protocol. `ftoverlay.cpp` was also compile-checked against `openvr.h` on a PC.

The status page (`http://<headset address>:8000`) prints per-step timings: straighten, follow, memory, edges, vivid, picture,
panel, left eye; pictures made versus pictures reaching the panel; head-tracking rate; and pairs of left/right frames out of step.

## Measured on the author's headset

- Cameras: about 30 usable frames per second; the two front cameras expose at the same instant (0.6 ms apart).
- Color model (256 input): about 200-240 ms per answer, 3.5-4 answers per second, with everything else running.
- Display: the first versions made only 8 pictures per second because the display work had no core of its own. With a core
  of its own, the panel hand-off in a helper thread and small image cuts: 28 pictures/s at 640 wide, 22-24 at 960 (one camera),
  24-26 at 960 with one view per eye.
- Outlines (FastSAM-s): about 340 ms on one fast core, 500-670 ms on the two slow cores.
- Heat: the hottest sensor read 60 C early in a session with audible fans; long sessions were not measured.

## Tried and abandoned (please do not retry without a new idea)

- **ncnn with Vulkan** crashed. **NPU access**: closed (Valve uses it for eye tracking). Everything is CPU with ONNX Runtime.
- A see-through color wash over the headset's own passthrough (`FT_TINT`): it does not line up.
- `SetOverlayRaw` for the main picture: it flickers (a Vulkan texture path is used instead).
- A memory that resets on change; brightening the input to fix the model's orange cast (the cast is **not** caused by dim
  pictures; white balance fixes it well enough).
- Small student colorization models trained on a PC GPU with the big DDColor as teacher (9,600 COCO photos, a U-Net from scratch
  and a MobileNetV2-based one): both came out as a sepia filter. It would need far more photos and probably another loss.

## Ideas for optimization and improvement

- Move the final color mixing (Lab to RGB, upscaling, side-by-side assembly) to the GPU inside `ftoverlay`.
- Interpolate head poses instead of taking the nearest 100 Hz sample; try `FT_POSE_LAG=0.01`.
- Add the delivery-evenness report described in `KNOWN_ISSUES.md` (stutter) and look at heat throttling over long sessions.
- A faster or better color model; or a model that predicts a few colors per region instead of per pixel.
- Make the memory position-aware (it is by direction only), and speed up the response to new or fast-moving things.
- Check the left eye's colors on hands, and the corner artifacts of the one-view-per-eye mode.
- Test on other headsets and SteamOS versions and report what breaks.
