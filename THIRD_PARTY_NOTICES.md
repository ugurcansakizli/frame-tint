# Third-party notices

Frame Tint itself is MIT licensed (see `LICENSE`). It contains or builds on the following. License texts are in
`third_party/licenses/` or next to the files.

## Passthrough PiP for Steam Frame (MIT, Copyright (c) 2026 Sippbox)

`src/driver/driver_frametint.cpp`, `src/common/xrservice.hpp`, `src/common/fd_protocol.hpp`, `src/ftgrab/ftgrab.c`,
`src/overlay/canvas.cpp`, `src/overlay/canvas.hpp`, `src/overlay/ftoverlay.cpp`, `share/input/*.json` and the launcher's
overall approach are adapted from that project (the camera-buffer lending technique, the SteamVR background-app setup,
Vulkan texture upload, the software canvas, the SteamVR Input action manifest). Full text: `third_party/licenses/Passthrough-PiP-MIT.txt`.

## OpenVR SDK headers (`third_party/openvr`)

Copyright (c) 2015, Valve Corporation. BSD 3-Clause License; full text in `third_party/openvr/LICENSE`.

## Khronos Vulkan headers (`third_party/vulkan`)

Copyright 2015-2026 The Khronos Group Inc. `SPDX-License-Identifier: Apache-2.0 OR MIT`. Apache-2.0 text in
`third_party/licenses/Vulkan-Headers-Apache-2.0.txt`. Source: https://github.com/KhronosGroup/Vulkan-Headers

## stb_truetype and stb_image (`third_party/stb`)

By Sean Barrett. Public domain (or MIT, at your option); the license text is at the end of each header.

## Atkinson Hyperlegible (`share/fonts`)

Copyright 2020 Braille Institute of America, Inc. SIL Open Font License 1.1; full text in
`third_party/licenses/AtkinsonHyperlegible-OFL.txt`.

## Models (included in `models/`)

- **DDColor** (https://github.com/piddnad/DDColor): the colorization model. Apache-2.0. `models/ddcolor-fast2-8bit-256.onnx` is a
  modified copy (the `paper_tiny` variant, exported to ONNX, 8-bit quantized, fixed 256x256 input, further tweaked with
  help from Claude Fable 5.2). Original copyright belongs to the DDColor authors; the license text is Apache-2.0.
- **FastSAM** (https://github.com/CASIA-IVA-Lab/FastSAM): the optional outline model. Several sources (Qualcomm AI Hub,
  Hugging Face model cards) list it as **AGPL-3.0** while at least one mirror says Apache-2.0. `models/fastsam-s.onnx` (an ONNX export of
  FastSAM-s) is included here; if it is AGPL-3.0 it stays under that license, separate from this repo's MIT code, and
  you must check its terms before redistributing it. Frame Tint runs without it (`FT_THINGS=0`; colors then follow edges only).

## Python packages (installed by `install.sh`, not included)

NumPy (BSD-3-Clause), OpenCV (Apache-2.0), ONNX Runtime (MIT).
