"""Camera tests render on the CPU rasterizer (lavapipe), so images do not depend on the GPU."""

import os

os.environ.setdefault("AUTONOMOUSIM_RENDER_ADAPTER", "software")
