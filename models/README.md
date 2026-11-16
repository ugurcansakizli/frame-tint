# Models

Both models are **included in this folder** and `install.sh` copies them to `~/.local/share/frametint/models/`
(licenses: see `THIRD_PARTY_NOTICES.md`). To use another model, put it in that folder. `bash scripts/get-models.sh`
still downloads anything listed in `models/models.txt` (file name, URL, sha256).

SHA-256:
```
d79f8d3bd22ebbac5a7e29d9a5b864eaeb70103007f22cda6a8b8e08dea75c5e  ddcolor-fast2-8bit-256.onnx
760992e0111916c5ce99e5757295cd16e5082f9e4abf301cc03e62a16f067d59  fastsam-s.onnx
```

## Color model (required)

Default file name: `ddcolor-fast2-8bit-256.onnx`. Choose another with the first argument of `live_server.py`.

A **DDColor** model (https://github.com/piddnad/DDColor, Apache-2.0), the `paper_tiny` variant exported to ONNX, made small and
fast enough for the headset's CPU: 8-bit quantized, with a fixed 256x256 input, then tweaked further by the author with
the help of Claude Fable 5.2.

What `live_server.py` expects of it:
- one input of shape `[1, 3, S, S]`, float32, values 0..1; S is read from the model (256 here). The image is the camera's gray picture
  copied into all three channels;
- one output whose first element is `[2, h, w]`: the Lab `a` and `b` channels (in Lab units, roughly -128..127), which are
  resized to the working size. Brightness is never taken from the model: it always comes from the live camera frame.

> Note: the author's export environment had `torch 2.15.0`, `onnx 1.23.2`, `onnxsim 0.8.1` (also `ncnn`, `mnn`, `pnnx` for experiments). Add the exact recipe that produced `ddcolor-fast2-8bit-256.onnx` from `ddcolor_paper_tiny` (export
> command, quantization settings), or a download link and hash in `models.txt`.

## Outline model (optional)

Default file name: `fastsam-s.onnx` (override with `FT_THINGS_MODEL`). FastSAM-s finds the outlines of things so each
thing gets one color. Without it (`FT_THINGS=0`, or the file missing) colors follow image edges only, which is a little less tidy.
FastSAM is listed as AGPL-3.0 by several sources: check the license before redistributing it.

> Expected layout (see `src/server/segmenter.py`): input `images` [1,3,256,320]; outputs are the detections and the 32 mask prototypes, as in the standard FastSAM-s ONNX export.
