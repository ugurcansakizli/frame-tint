# Architecture

Everything runs on the Steam Frame (SteamOS, aarch64, Snapdragon 8 Gen 3, 8 CPU cores: 0-1 slow, 2-7 fast).

```
 XRService (Valve, owns the cameras)                 vrserver (SteamVR)
   captures into dma-buf buffers                       loads driver_frametint.so
            |                                                |
            |   pidfd_getfd (allowed: vrserver is XRService's parent)
            |                                                v
            |                                  cameras.sock (SCM_RIGHTS)
            v                                                |
        [camera buffers] <---- fds ---- ftgrab (C) ----------+   one process per camera
                                           | gray frames on stdout ("FTG1")
                                           v
                          live_server.py (Python, the main program)
                          rectify -> color model -> memory/follow/snap -> compose
                                           | picture slot number ("FTS1", pixels in shared memory)
                                           v
                          ftoverlay (C++, an OpenVR background app) --> SteamVR overlay in the headset
                                           |  stdout: head pose "FTP1", ack "FTA1", on/off "FTT1", quit "FTQ1"
                                           +--> back to live_server.py
```

## Components

| Path | What it is |
|---|---|
| `src/driver/driver_frametint.cpp` | A device-less SteamVR driver loaded into `vrserver`. Lends the tracking cameras' capture buffers to `ftgrab`. |
| `src/ftgrab/ftgrab.c` | Asks the driver for buffers, maps them, picks the long-exposure frames and streams them as gray images. |
| `src/overlay/ftoverlay.cpp` | OpenVR background app: shows the picture as an overlay, publishes head pose, owns the dashboard panel and the controller toggle. |
| `src/overlay/canvas.*`, `dash_ui.hpp` | A tiny software rasterizer with TrueType text (from Passthrough PiP) and the dashboard panel's drawing. |
| `src/server/live_server.py` | The main program: all image processing, plus a small status web page. |
| `launcher/` | Start/stop scripts and the desktop entry. |
| `share/` | Resources used by `ftoverlay`: SteamVR Input action manifest and default binding, fonts, icon. |

## Getting the camera frames without root

The tracking cameras belong to Valve's `XRService`, which captures into shared dma-buf buffers. Reading them needs a
copy of XRService's file descriptors (`pidfd_getfd`). With Yama `ptrace_scope = 1` the kernel only allows that to an
ancestor, and XRService runs as a child of `vrserver`. So a driver loaded into `vrserver` does the copying and hands the
descriptors to `ftgrab` over a `SOCK_SEQPACKET` unix socket (`/run/user/<uid>/frametint/cameras.sock`, `SCM_RIGHTS`).
This is the method of Passthrough PiP for Steam Frame; `driver_frametint.cpp` is adapted from its driver.

The driver lends narrowly: only to the same user, only if the caller's `/proc/<pid>/exe` is the installed `ftgrab`
(`<root>/bin/ftgrab`, the root being where the driver library itself lives), and only the capture buffers of tracking
cameras, which it finds itself (XRService's log names each camera's `/dev/video` node; `VIDIOC_QUERYBUF` lists the buffers).
Messages are defined in `src/common/fd_protocol.hpp` (C++) and `fd_protocol_c.h` (C); the two must keep identical sizes.

Camera indices (from `/persist/xrservice.json`): 0 = `slam_right`, 1 = `slam_left` (both 1056x1024 fisheye, Kannala-Brandt
model, fx about 327, front-facing, 106 mm apart, each turned about 29 degrees outward and 21-24 degrees down),
2 = `upper_left`, 3 = `upper_right` (640x480). The cameras alternate a long exposure (a normal picture) with a short one
(nearly black, for controller tracking), so about 30 usable frames per second. `ftgrab` tells them apart by brightness
and frame-number parity, and works out the true row pitch of the buffer (it is not always what the camera reports) by
choosing the candidate that makes neighbouring rows most alike. The cameras pause when the headset is not worn.

## Stream and message formats

**`ftgrab` stdout**: repeated records: 24-byte header (`"FTG1"`, `uint32` width, `uint32` height, `uint32` frame
number, `double` time as `CLOCK_MONOTONIC` seconds when the grabber saw the frame), then `width*height` gray bytes.

**`ftoverlay` stdin** (little endian):
- `"FTV1"`, `uint32 w`, `uint32 h`, then `w*h*4` RGBA bytes.
- `"FTV2"`, `w`, `h`, `uint32 pinned`, 12 `float` (3x4 head matrix the picture belongs to), then RGBA bytes. With `pinned`
  the panel is placed in the room where the head was at that moment, so SteamVR keeps it lined up with the room between pictures.
- `"FTS1"`, `w`, `h`, `pinned`, 12 floats, `uint32 slot`: like FTV2 but the pixels are slot `slot` of the shared-memory
  file named by `FT_SHM` (`/dev/shm/frametint-view`, 3 slots), so no pixels cross the pipe.

**`ftoverlay` stdout** (with the `poses` argument): fixed 52-byte records: `"FTP1"` + 12 floats (head pose, about 100 Hz),
`"FTA1"` (a picture has been handed to SteamVR; the sender waits for it before reusing a slot), `"FTT1"` + float
(1 = switched on, 0 = off, from the controller or dashboard), `"FTQ1"` (Quit pressed in the dashboard panel), and with
`gaze` `"FTE1"` + 3 floats after each pose (valid, then where the eyes look relative to the head as x/-z and y/-z).

`ftoverlay` arguments: `<panel width m> <distance m> <opacity> [poses] [sbs] [gaze]`; environment `FT_TEXEL`, `FT_SHM`, `FT_SHARE`.

## Eye tracking (`FT_FOVEA=1`)

A background app gets the gaze from `IVRSystem::GetEyeTrackedFoveationCenter` (per-eye NDC; undone through each eye's
projection matrix, NDC y up). It answers about 100 times a second on the Frame. The SteamVR Input route (an
`eyetracking` action bound to `/user/eyes_ext/input/gaze_ext`) loads without errors but returns no data to a
background app. Every `FT_FOVEA_EVERY`-th color answer runs the model on a `FT_FOVEA_DEG` square of the straightened
picture centred on the gaze (moved to the frame's head turn); that answer is laid with a soft edge over the newest
whole-view answer (turned to the same frame) and written to the color memory with double weight inside the area.
Colouring only the looked-at thing and leaving the rest grey was tried and dropped: losing the periphery is worse.

## The color pipeline (`live_server.py`)

Threads, each pinned to its own cores (configurable):

1. **reader** (and **reader_left**): reads `ftgrab` output, keeps the newest frame, attaches the head pose nearest to
   `now - FT_POSE_LAG` (a frame reaches Python a little after it was exposed).
2. **colorer** (cores 4-7): rectifies the camera image to a 115-degree view, runs the color model (a gray image in, Lab a/b
   out, 256x256 input), then removes part of the picture's overall color cast (`FT_BALANCE`, white balance by subtracting the
   median a/b of the answer, smoothed over answers). About 3.5-4 answers per second on the author's unit.
3. **Color memory** (`Memory`): colors stored by direction in the room as a 1024x512 equirectangular map (a, b,
   brightness, confidence), written with the head pose of each answer. Where a spot no longer looks like what was
   remembered (a hand, a moved thing), the newest answer replaces it.
4. **Follower**: carries the newest answer's colors onto the newest camera frame: undoes the head turn since the answer's frame
   with a homography, then follows what else moved with DIS optical flow.
5. **composer** (core 2): per camera frame, blends memory and followed colors, snaps colors to edges (watershed patches,
   guided filter), optionally to whole things (**outliner**, FastSAM-s outlines a few times a second: one color per thing),
   boosts saturation (`FT_VIVID`, a `tanh` curve), converts 8-bit Lab to RGB with a lookup table for brightness and sends
   the picture to `ftoverlay`. Colors are worked out at 320x240 and upscaled; brightness is always the full-size newest frame.
6. **LeftEye** (core 3, with `FT_STEREO=1`): the left camera, rectified to the same forward view, borrows the right eye's
   colors from one picture ago moved with DIS flow between the left and right views, then a guided filter. Left and right
   frames are paired by the grabbers' timestamps (within 12 ms). The two views go side by side, each shifted by
   `(IPD/2) / panel width` of the picture width so far things are straight ahead of each eye.

Conventions (verified on real pictures): a rectified pixel `(u, v)` maps to the head direction `((u-cx)/f, -(v-cy)/f, -1)`,
world = `R * d`. The left camera's direction comes from `axes(cad camera 0) x axes(cameras[1].extrinsics)`; the `cad` entry
for camera 1 is about 3.3 degrees off and must not be used. The camera has about 1028 pixels across the 115-degree view,
so a 960-wide picture is close to its real detail.

## Display

`ftoverlay` joins SteamVR as a **background app** (`VRApplication_Background`): unlike an overlay app it never starts
SteamVR itself (a `vrserver` started outside the Frame's gamescope session keeps the session from starting; this is a
finding from Passthrough PiP). Pictures are handed over as Vulkan textures, two used in turn, so the panel never shows a
half-updated picture. `SetOverlayRaw` flickers and is only used for the small dashboard panel. `VROverlayFlags_SideBySide_Parallel`
works on the Frame (left half to the left eye); the texel aspect must be 1.0 (SteamVR already halves the shape; 0.5 squashes it
and makes trapezoids when the head tilts).

The **dashboard panel** is a dashboard overlay with mouse input, drawn by `canvas.cpp` (`dash_ui.hpp` has the layout).
The **controller toggle** is a SteamVR Input action (`share/input/actions.json`; default binding: double-tap `View` on the
left controller). Both switch the picture on and off through the same function and tell `live_server.py` with `FTT1`, which
pauses all processing threads.
