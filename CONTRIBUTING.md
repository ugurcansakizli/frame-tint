# Contributing

Thanks for looking. The most useful contributions right now:

1. **Test reports.** Does it install and run on your Steam Frame? Open an issue with your SteamOS and SteamVR versions, the
   end of `~/.local/state/frametint/frametint.log`, and the status page's line (camera rate, answers per second, pictures made
   versus reaching the panel).
2. **The three open bugs** in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md): the SteamVR window going behind the picture, the dock
   icon placeholder, and the stutter on fast head turns.
3. **Speed and heat**: see the optimization ideas in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Please measure before and after
   with the status page, and repeat each run: averages hide hiccups.

Ground rules:

- Everything must keep running **on the headset** (no PC offload), and without `sudo`.
- Keep changes small and explain what you measured. Say what you tested on and what you did not.
- Do not commit recordings, calibration files or model weights (`.gitignore` blocks the usual ones): they contain pictures of
  your room, your headset's serial numbers, or have their own licenses.
- By contributing you agree your contribution is under the project's MIT license.
