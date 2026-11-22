#!/usr/bin/env bash
# Checks everything physics-studio needs and prints how to fix what is missing. Tested on Linux only.
# Defaults can be changed with environment variables:
#   PHYSICS_STUDIO_PYTHON        Python with mujoco, numpy, pillow, imageio-ffmpeg (default: ./.venv/bin/python, else python3)
#   MUJOCO_GL                    offscreen OpenGL backend: egl or osmesa (default egl)
#   KOKORO_HOME                  folder with the Kokoro model files and its venv (default ~/kokoro)
#   PHYSICS_STUDIO_VOICE_PYTHON  Python that has kokoro-onnx (default $KOKORO_HOME/.venv/bin/python, else python3)
#   FF                           ffmpeg binary (default: ffmpeg on PATH, else the one inside imageio-ffmpeg)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PHYSICS_STUDIO_PYTHON:-}" ]; then PY="$PHYSICS_STUDIO_PYTHON"
elif [ -x "$R/.venv/bin/python" ]; then PY="$R/.venv/bin/python"
else PY="$(command -v python3 || true)"; fi

if [ -z "$PY" ] || ! "$PY" -I -c "import mujoco, numpy, PIL" 2>/dev/null; then
  echo "MISS python with mujoco, numpy and pillow"
  echo "       cd \"$R\" && python3.12 -m venv .venv && .venv/bin/pip install mujoco numpy pillow imageio-ffmpeg pytest"
  echo "       (or with uv: uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python mujoco numpy pillow imageio-ffmpeg pytest)"
  echo "something is missing: fix the MISS lines above and run this again"
  exit 1
fi
exec "$PY" "$R/engine/cli.py" check "$@"
