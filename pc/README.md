# Optional PC helper

Frame Tint normally runs entirely on the headset. If a PC with an NVIDIA graphics card is on the same network, it can
run the **large** DDColor model instead of the headset's small one: more realistic colors, about twice as many color
answers per second, and a cooler headset. When the PC does not answer, the headset falls back to its own model by
itself and tries the PC again every 5 seconds.

Measured with an RTX 4060 laptop GPU: about 115 ms per picture on the PC (grain removal 39 ms on the GPU, model 70 ms in
16-bit), 120-130 ms round trip from the headset, about 7.8 color answers per second (headset model alone: about 3.3).

## Setup on the PC

1. Python with PyTorch (CUDA), OpenCV and NumPy.
2. The DDColor code: `git clone https://github.com/piddnad/DDColor`
3. The large model: `config.json` and `pytorch_model.bin` from https://huggingface.co/piddnad/ddcolor_modelscope (Apache-2.0)
4. Start it, and leave it running:
   ```
   python colorize_server.py --ddcolor <DDColor folder> --weights <folder with the two files>
   ```
   Windows asks once whether Python may accept connections on private networks: allow it.

## Setup on the headset

In `~/.config/frametint/frametint.conf`, add the PC's address as the headset sees it, then restart Frame Tint:
```
FT_REMOTE=<PC address>:8765
```
If the PC is connected to the headset's own Wi-Fi hotspot, that address (10.35.78.x) is the fastest link.
The status page shows where the colors come from (`remote`). `curl "localhost:8000/set?remote=0"` on the headset switches
back to the headset model while running (`remote=1` returns to the PC).

## What crosses the network

The headset sends a 512x512 black-and-white JPEG of its camera view (about 60 KB) and gets back 128x128 colors (32 KB).
Pictures go only to the PC you name; the server keeps nothing and only accepts connections from private (home)
network addresses (`--allow` to change).

`nlmeans_gpu.py` is the grain removal the large model needs (the tracking camera's picture is grainy), as a PyTorch GPU
version of OpenCV's `fastNlMeansDenoising` (within 0.25 grey levels of it).
