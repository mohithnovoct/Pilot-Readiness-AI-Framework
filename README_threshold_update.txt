The requested adjustable threshold functionality has been completely implemented.

I've added the following:
1. Modified `streaming_demo.py` to calculate threshold based on standard deviation (1 - alpha, where alpha is 0.05 default).
2. Added `set_threshold` socket event in `streaming_demo.py` to allow the threshold to be adjusted dynamically during the stream simulation.
3. Added styling in `streaming.html` for a nice looking slider control (`<input type="range">`), custom webkit aesthetics included.
4. Added the threshold slider to the header controls, positioned nicely next to the speed dropdown, with real-time value display update using inline `oninput`.
5. Created handling on `setThreshold` JS function which pushes socket events.
6. Allowed dynamic syncing so if Server overrides threshold it's updated correctly in UI slider state.

Everything can be tested smoothly.
