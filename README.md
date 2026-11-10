# nano-vla: LIBERO-Spatial playback

Python 3.10 environment managed by uv. Plays recorded **simulator states**, not
policy actions, in the native interactive MuJoCo viewer. No policy, checkpoint,
PyTorch, or training dependencies required.

## Setup (already completed on this machine)

```bash
# Required on a fresh checkout: upstream assets are not committed here.
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv sync --locked
```

The local editable LIBERO source retains upstream assets; its namespace packaging
is incomplete, so playback locates the source assets directly. Avoid installing
upstream's full requirements: they include unnecessary training dependencies.

## Download

```bash
uv run python scripts/download_libero.py --list
uv run python scripts/download_libero.py --task between  # one task, ~509 MB
uv run python scripts/download_libero.py                 # all 10 tasks, ~6.24 GB
```

Official source: https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
Downloads go to `datasets/libero_spatial/`. Interrupted `.part` downloads resume
on rerun. Validation is by file size, not cryptographic checksum.

## Replay in GUI

```bash
# Ready now: task "between", first episode.
uv run python scripts/replay_libero.py --task between --episode 0

# Selected episodes; names demo_0, demo_1 etc. also work.
uv run python scripts/replay_libero.py --task between --episodes 0,3,7

# All episodes of one task, at double speed.
uv run python scripts/replay_libero.py --task between --episode all --speed 2

# All downloaded tasks and all episodes.
uv run python scripts/replay_libero.py --task all --episode all

# Inspect downloaded tasks/episodes without opening a viewer.
uv run python scripts/replay_libero.py --list
```

A unique filename substring or full HDF5 filename selects a task. Default is the
first downloaded task and episode 0. Each episode opens a new viewer; closing the
window stops the entire run. Native MuJoCo viewer controls allow camera and
visualization interaction; initial camera is `agentview`. Run from the repo root;
`--data-dir` overrides the dataset directory. GUI needs a working desktop/display.

Playback restores the episode XML (repairing historical asset paths) and each
recorded state, then runs MuJoCo forward kinematics/dynamics calculations. It does
not execute actions or advance physics. Default display rate is 20 FPS, adjustable
with `--fps` / `--speed`.

## Validation

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/replay_libero.py --task between --headless --max-frames 100
```

Verified here: nine unit tests; all 50 episodes / 5,068 states of the downloaded
"between" task restored headlessly; native viewer launched and exited successfully
in a 60-frame GUI smoke test. Other nine task files have not yet been downloaded
or runtime-tested. The full suite downloader and task selection are available.

Core versions: MuJoCo 2.3.7, robosuite 1.4.0, NumPy 1.22.4, Numba 0.57.1,
h5py 3.10.0. `uv.lock` pins all dependencies. robosuite's missing private macros
warning is harmless for playback.
