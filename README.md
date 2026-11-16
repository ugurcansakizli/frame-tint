# Frame Tint

Software color passthrough for the Valve Steam Frame, running entirely on the headset.

The Steam Frame's built-in passthrough is black and white. Frame Tint reads the headset's two front tracking cameras,
guesses colors for them with a neural colorization model (DDColor) on the headset's own CPU, remembers those colors by
direction in the room using head tracking, and shows the result as an overlay in SteamVR. No PC and no extra camera
hardware are involved.

> **Status: experimental prototype.** It works on the author's headset, with known rough edges
> (see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)). The colors are plausible guesses made by a neural network, **not
> true colors**: a red wall can come out orange, and new or fast-moving things stay grey until the next color answer.
> Frame Tint is unofficial and not affiliated with Valve. It relies on undocumented SteamVR and SteamOS details, so an
> update may break it.

## What you get

- A color view of the room, either one flat picture or (experimental) one view per eye from the two front cameras.
- A **Frame Tint panel in the SteamVR dashboard** with a Switch ON/OFF button and a Quit button.
- A controller shortcut: double-tap **View** on the left controller to hide or show it (it does not work while the SteamVR
  menu is open; use the dashboard panel then). Hiding it also pauses the processing, which cools the headset.
- **No `sudo` needed.** A small SteamVR driver lends the camera buffers to the grabber (same technique as
  Passthrough PiP for Steam Frame, see Credits).
- A launcher entry, "Frame Tint", in SteamVR's Launch Program list and the desktop app menu.

## Requirements

- A Steam Frame running SteamOS, with SteamVR. Everything is built and run on the headset (aarch64).
- The headset's own calibration file `/persist/xrservice.json` (it is there on a normal unit).
- Internet access on the headset during installation (Python packages, models).
- A way to type commands on the headset: Desktop Mode's terminal, or `ssh steamos@<headset>`.
- Tools used by the installer: `g++`, `gcc`, `python3` (with `venv`) and `curl`. The installer checks for them and stops with a message if one is missing.

Expect heavy CPU use and a warm headset. The color model uses four of the eight cores.

## Install

On the headset:

```
git clone https://github.com/ugurcansakizli/frame-tint.git
cd frame-tint
bash install.sh
```

The installer builds three small programs, creates a Python environment, registers the camera driver with SteamVR and
adds the launcher. It writes only inside your home folder (`~/.local/share/frametint`, `~/.local/bin`,
`~/.config/frametint`). Then:

1. **Restart SteamVR once**, so it loads the camera driver (this reloads the whole headset session, like a quick reboot):
   ```
   XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus systemctl --user restart steamvr.service
   ```
2. Put the headset on and start **Frame Tint** from SteamVR's *Launch Program* list or the desktop app menu.

**The two models are included** in `models/` (about 108 MB together) and copied by `install.sh`. See [models/README.md](models/README.md) for what they are and their licenses.

To remove everything: `bash uninstall.sh`.

## Use

- The color view appears in front of you a few seconds after start (the first color answers take a moment).
- Open the SteamVR dashboard and click the **Frame Tint** panel: **Switch OFF/ON** hides or shows the picture and pauses
  or resumes the work; **Quit Frame Tint** ends the program.
- A status page is served at `http://<headset address>:8000` (open it from a PC on the same network). It shows camera
  rate, color answers per second, timings of each step and whether the two eyes are in step.
- Settings live in `~/.config/frametint/frametint.conf` (one `NAME=value` per line). They apply the next time you start
  Frame Tint. The log is `~/.local/state/frametint/frametint.log`.

### Settings

| Setting | Meaning | Default |
|---|---|---|
| `FT_STEREO` | `1`: one view per eye (experimental). `0`: one flat picture from the right camera (more reliable) | shipped config: `0` |
| `FT_RES` | width of the picture in pixels; 960 is near the camera's real detail, 640 is faster | 640 (shipped config: 960) |
| `FT_SCALE` | size of the panel: 1 = life size (fills most of the view), 0.5 = half | 0.5 (shipped config: 1) |
| `FT_VIVID` | how much stronger than the model's own answer the colors are shown (1 = as is) | 1.8 (shipped config: 1.2) |
| `FT_BALANCE` | white balance, 0..1: how much of the picture's overall color cast is removed | 0.5 (shipped config: 0.3) |
| `FT_ALPHA` | opacity of the panel, 0..1 | 1 |
| `FT_IPD` | distance between your eyes in metres (used with `FT_STEREO=1`) | 0.063 |
| `FT_THINGS` / `FT_MEMORY` / `FT_SNAP` / `FT_PIN` | `0` switches off: outline model / color memory / edge snapping / pinning the picture to the room | on |
| `FT_CORES`, `FT_DISPLAY_CORES`, `FT_LEFT_CORES`, `FT_THINGS_CORES` | which CPU cores each part may use (the headset has 8: cores 2-7 are the fast ones) | `4,5,6,7` / `2` / `3` / `3` (`0,1` with stereo) |
| `FT_POSE_LAG` | seconds between a camera exposure and the head pose assumed for it | 0.02 |

The full list, with explanations, is at the top of [`src/server/live_server.py`](src/server/live_server.py).

## Troubleshooting

- **Nothing appears / the log says the camera driver is not loaded.** Restart SteamVR (step 1 above), with the headset on.
- **"camera paused (the headset thinks nobody is wearing it)".** The cameras only run while the headset is worn.
- **"overlay: could not join SteamVR".** SteamVR must be running first; start Frame Tint from inside SteamVR.
- **A leftover process after a crash.** Run `~/.local/bin/frametint-stop`.
- Still stuck? Open an issue with the end of `~/.local/state/frametint/frametint.log`, your SteamOS and SteamVR versions
  and what the status page says.

## How it works, and how to help

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the pieces, the data flow, the file protocols between them.
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): open bugs with everything measured and tried so far. Good places to start.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): building, running from a checkout, measurements, dead ends, optimization ideas.
- [`tools/`](tools/README.md): recording and measuring scripts (camera check, benchmark, stereo test).
- [CONTRIBUTING.md](CONTRIBUTING.md)

The project's one hard rule so far: everything runs **on the headset** (no PC offload).

## Credits

- **Passthrough PiP for Steam Frame** by Sippbox (MIT; [github.com/Sippbox/frame-passthrough-pip](https://github.com/Sippbox/frame-passthrough-pip)): the technique of borrowing the
  camera buffers through a driver inside `vrserver`, plus parts of the driver, the overlay setup and the software
  canvas used for the dashboard panel are adapted from it.
- **DDColor** (Apache-2.0), the colorization model. **FastSAM** (listed as AGPL-3.0), the outline model. Both are included, converted, in `models/`.
- OpenCV, ONNX Runtime, NumPy; the OpenVR SDK headers (BSD-3-Clause), Khronos Vulkan headers (Apache-2.0 OR MIT), stb
  (public domain), Atkinson Hyperlegible (SIL OFL).

Details and license texts: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Created by [Uğurcan Sakızlı](https://github.com/ugurcansakizli), developed together with Claude (Anthropic), which wrote much of the code and documentation.

## License

[MIT](LICENSE).
