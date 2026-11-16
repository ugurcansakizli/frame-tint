# Known issues

Three bugs are open. Each lists what is known, so nobody has to start from scratch. Measurements are from the author's
single headset.

## 1. The SteamVR window disappears behind the Frame Tint picture

**Symptom.** Open the SteamVR dashboard/menu and push the window away from you. Past a certain distance it vanishes
behind the color picture.

**Likely cause (from the OpenVR header; the dashboard's own sort order was never checked).** Overlays with the same sort order are drawn back to front by distance from the head.
With `FT_SCALE=1` the picture is a 4.7 m wide panel 1.5 m away (`DIST` in `live_server.py`), so a window pushed beyond about 1.5 m counts
as farther and is drawn first, under the picture. Sort order cannot help directly: `SetOverlaySortOrder` takes an
unsigned number whose default is 0, so the picture cannot be put below an overlay that is also at 0, and the sort order of
SteamVR's own dashboard is unknown.

**A fix that was built but not adopted.** While the dashboard is open (`IVROverlay::IsDashboardVisible()`), move the picture
far away (about 20 m, width scaled by the same factor so it looks the same) so it always sorts behind. It compiled but was
never run on a headset: the author decided against the approach. Its downsides: the per-eye shift (`EYE_SHIFT`) is computed
for 1.5 m, so depth is slightly off while the window is open, and a window pushed beyond 20 m would still go behind.
Other ideas: hide the picture while the dashboard is open; find an overlay property or flag that draws ours underneath
regardless of distance.

## 2. The dock icon of the dashboard panel is a grey question mark

**Symptom.** The Frame Tint entry appears in the SteamVR dashboard dock and works (switch, quit), but its icon is SteamVR's
placeholder.

**Tried, without effect:**
- `SetOverlayFromFile(thumbnail, "<share>/icon.png")` returns `VROverlayError_None`. The file is a 1024x1024, 8-bit RGBA PNG,
  the same format as Passthrough PiP's icon (it was first 128x128, which showed the same placeholder).
- Setting the icon from memory with `SetOverlayRaw` on the thumbnail handle (the entry could not be found in the dock at all; cause unclear).
- Re-applying the icon 4 seconds after start and every time the panel is shown (`reapplyIcon` in `ftoverlay.cpp`).
- A new overlay key (`frametint.dashboard.v2`), in case SteamVR cached the first entry.
- `GetOverlayTextureSize(thumbnail)` returns `VROverlayError_InvalidHandle`. This may be meaningless for thumbnail overlays,
  as the header says some calls do not apply to them.

**Ideas.** Upload the icon as a Vulkan texture with `SetOverlayTexture` on the thumbnail (the main picture already uses that
path); compare byte for byte what Passthrough PiP does differently; check whether SteamVR stores dock icons somewhere under
`~/.config/openvr` or `~/.local/share/Steam/config` and whether that needs clearing; check whether PiP's own icon shows on
your headset (that separates "file problem" from "environment problem").

## 3. A slight stutter when turning the head fast

**Symptom.** When the head turns fast (120-370 degrees per second) the whole picture sometimes lags and jumps as one piece,
about five times a second. It comes and goes between runs.

**It is not** tied to one camera versus two: run back to back, the two modes swapped which one stuttered.

**Measured and ruled out:** pictures lost before the panel; head tracking rate and arrival (steady 100 Hz); a second grabber
running; the left camera's stream; left/right alignment, still and moving (about 1.1 px); time offset between the cameras
(0.6 ms; they expose at the same instant, global-shutter-like, and frames arrive 0-2 ms apart in no fixed order); left/right
pairing; frame arrival jitter in Python (under 1 ms).

**Not yet measured:**
- evenness of picture delivery and the time SteamVR takes to accept a picture. A report of this (gaps between pictures, hand-over
  time, head-tracking health, every 4 seconds) was written in an earlier session, but it is **not in this repository**.
  Adding it to `ftoverlay.cpp` is a good first step;
- head tracking health flags, and heat throttling (the hottest sensor was 60 C early in a session; long sessions were never measured);
- which cores unpinned programs land on.

**Hints.** Poses are chosen as "nearest of the 100 Hz samples" and not interpolated; a recording suggests the best
`FT_POSE_LAG` is about 0.01, not the default 0.02. Averages such as pictures per second hide hiccups: repeat each run before
trusting a comparison.

## Other limitations

- Colors are plausible, not correct; white balance is a single global correction.
- New or fast-moving things stay grey until the next color answer (about every 250 ms). Memory is by direction, not by position.
- With one view per eye (`FT_STEREO=1`): depth is exaggerated (the cameras are 106 mm apart), each eye has a black corner (a
  different corner per eye), and the left eye's colors are borrowed from the right eye's, not checked closely on hands.
- CPU load and heat are high. The final color mixing could move to the graphics chip inside `ftoverlay`.
- The camera driver and `XRService`'s log format are undocumented; a SteamOS or SteamVR update can break them.
- `FT_TINT` (a see-through color wash over the headset's own view) is abandoned: it does not line up. It is still in the code.
