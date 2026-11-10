#!/usr/bin/env python3
"""Replay recorded LIBERO states, not actions, in the native MuJoCo viewer.

No benchmark / environment construction (or torch) is needed. Each episode's
model_file is authoritative; observations and actions are never loaded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import importlib.util
import json
import math
import os
import random
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

DEFAULT_DATA_DIR = Path("datasets/libero_spatial")


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def episode_selection(value):
    if value == "all":
        return None
    result = []
    for part in value.split(","):
        suffix = part.strip().removeprefix("demo_")
        if not suffix.isdigit():
            raise argparse.ArgumentTypeError("use all, 0, demo_0, or comma-separated episode numbers")
        name = f"demo_{int(suffix)}"
        if name not in result:
            result.append(name)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", help="unique filename substring, HDF5 path, or 'all' (default: first task)")
    parser.add_argument("--episode", "--episodes", type=episode_selection, default=["demo_0"],
                        help="episode number/name, comma-separated numbers, or all (default: 0)")
    parser.add_argument("--no-random", action="store_true",
                        help="disable default task/episode shuffling; preserve sequential or supplied order")
    parser.add_argument("--list", action="store_true", help="list selected tasks and episode lengths, then exit")
    parser.add_argument("--fps", type=positive_float, default=20.0, help="recorded frame rate (default: 20)")
    parser.add_argument("--speed", type=positive_float, default=1.0, help="playback multiplier")
    parser.add_argument("--headless", action="store_true", help="restore states without creating a viewer or renderer")
    parser.add_argument("--max-frames", type=positive_int, help="total frame budget across selected episodes; headless default: 100")
    return parser


def discover_tasks(data_dir, task=None):
    if task and task != "all" and Path(task).is_file():
        return [Path(task).resolve()]
    files = sorted(p for p in Path(data_dir).glob("*.hdf5") if p.is_file())
    if not files:
        raise ValueError(f"No .hdf5 tasks in {data_dir}; run scripts/download_libero.py first")
    if task == "all":
        return files
    if task:
        matches = [p for p in files if task.lower() in p.name.lower()]
        if len(matches) != 1:
            raise ValueError(f"Task {task!r} matched {len(matches)} files; use --list or a more specific name")
        return matches
    return files[:1]


def episode_names(data, selected=None):
    names = sorted((name for name in data if name.startswith("demo_")),
                   key=lambda name: (int(name[5:]) if name[5:].isdigit() else math.inf, name))
    if selected is not None:
        missing = set(selected) - set(names)
        if missing:
            raise ValueError(f"Episodes not found: {', '.join(sorted(missing))}")
        names = list(selected)
    return names


def playback_order(items, no_random=False):
    ordered = list(items)
    if not no_random:
        random.shuffle(ordered)
    return ordered


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def libero_root():
    # Upstream's namespace layout is not included in its built wheel; use
    # the checked-out assets directly without importing the training package.
    checkout = Path(__file__).resolve().parents[1] / "third_party" / "LIBERO" / "libero" / "libero"
    if (checkout / "assets").is_dir():
        return checkout
    # Looking up the outer namespace does not execute libero.libero.__init__.
    spec = importlib.util.find_spec("libero")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            root = Path(location) / "libero"
            if (root / "assets").is_dir():
                return root.resolve()
    raise ValueError("LIBERO assets not found; install third_party/LIBERO with pip install -e third_party/LIBERO --no-deps")


@contextmanager
def isolated_libero_config(root, data_dir):
    """Preseed before any LIBERO import; never read/write ~/.libero or prompt."""
    previous = os.environ.get("LIBERO_CONFIG_PATH")
    with tempfile.TemporaryDirectory(prefix="nano-vla-libero-") as directory:
        paths = {"benchmark_root": str(root), "assets": str(root / "assets"),
                 "bddl_files": str(root / "bddl_files"), "init_states": str(root / "init_files"),
                 "datasets": str(Path(data_dir).resolve())}
        # JSON is valid YAML, and avoids a YAML dependency for configuration.
        (Path(directory) / "config.yaml").write_text(json.dumps(paths), encoding="utf-8")
        os.environ["LIBERO_CONFIG_PATH"] = directory
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("LIBERO_CONFIG_PATH", None)
            else:
                os.environ["LIBERO_CONFIG_PATH"] = previous


def repair_model_xml(xml, robosuite_root, root):
    """Port LIBERO postprocess_model_xml path repair without its torch import.

    Unlike envs.utils' cwd-dependent demo_generation branch, LIBERO assets are
    anchored at the installed package. Preserve the entire recorded model.
    """
    tree = ET.fromstring(text(xml))
    for element in tree.iter():
        filename = element.get("file")
        if not filename:
            continue
        parts = filename.replace("\\", "/").split("/")
        if "robosuite" in parts:
            index = len(parts) - 1 - parts[::-1].index("robosuite")
            path = Path(robosuite_root).joinpath(*parts[index + 1:])
        elif "assets" in parts:
            index = parts.index("assets")
            path = root.joinpath(*parts[index:])
        else:
            path = Path(filename)
        if not path.is_file():
            raise ValueError(f"Model asset not found: {filename} (resolved to {path})")
        element.set("file", str(path.resolve()))
    return ET.tostring(tree, encoding="unicode")


def replay(args, tasks):
    import h5py
    import robosuite
    from robosuite.utils.binding_utils import MjSim

    root = libero_root()
    budget = args.max_frames or (100 if args.headless else math.inf)
    count = 0
    from threading import Event
    quit_requested = Event()

    def on_key(keycode):
        # GLFW uses uppercase key codes regardless of Shift state.
        if keycode in (ord("Q"), ord("q")):
            quit_requested.set()

    with isolated_libero_config(root, args.data_dir):
        for task in playback_order(tasks, args.no_random):
            with h5py.File(task, "r") as file:
                if "data" not in file:
                    raise ValueError(f"{task}: missing data group")
                data = file["data"]
                for name in playback_order(episode_names(data, args.episode), args.no_random):
                    episode = data[name]
                    if "states" not in episode:
                        raise ValueError(f"{task.name}/{name}: missing states")
                    states = episode["states"]
                    if not len(states):
                        print(f"Skipping empty episode {task.name}/{name}", file=sys.stderr)
                        continue
                    if "model_file" not in episode.attrs:
                        raise ValueError(f"{task.name}/{name}: missing model_file XML")
                    xml = repair_model_xml(episode.attrs["model_file"], Path(robosuite.__file__).parent, root)
                    sim = MjSim.from_xml_string(xml)
                    try:
                        sim.reset()
                        expected = sim.get_state().flatten().size
                        if states.ndim != 2 or states.shape[1] != expected:
                            raise ValueError(f"{task.name}/{name}: states shape {states.shape}; expected (*, {expected})")
                        sim.set_state_from_flattened(states[0])
                        sim.forward()
                        if args.headless:
                            context = nullcontext(None)
                        else:
                            import mujoco.viewer
                            context = mujoco.viewer.launch_passive(
                                sim.model._model, sim.data._data, key_callback=on_key
                            )
                        print(f"Playing {task.name}/{name}: {len(states)} recorded states", flush=True)
                        with context as viewer:
                            if viewer is not None:
                                import mujoco
                                # Match LIBERO/robosuite: collision proxies (group 0)
                                # are often brightly colored and overlap visual meshes.
                                with viewer.lock():
                                    viewer.opt.geomgroup[0] = 0
                                    viewer.opt.geomgroup[1] = 1
                                camera_id = mujoco.mj_name2id(sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
                                if camera_id >= 0:
                                    # Start near agentview, but use a free camera so
                                    # mouse/trackpad navigation remains available.
                                    import numpy as np
                                    rotation = sim.data._data.cam_xmat[camera_id].reshape(3, 3)
                                    forward = -rotation[:, 2]
                                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                                    viewer.cam.fixedcamid = -1
                                    viewer.cam.distance = 1.0
                                    viewer.cam.lookat[:] = sim.data._data.cam_xpos[camera_id] + forward
                                    viewer.cam.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
                                    viewer.cam.elevation = np.degrees(np.arcsin(np.clip(forward[2], -1, 1)))
                            deadline = time.monotonic()
                            for state in states:
                                if quit_requested.is_set() or (viewer is not None and not viewer.is_running()):
                                    return count
                                with viewer.lock() if viewer is not None else nullcontext():
                                    sim.set_state_from_flattened(state)
                                    sim.forward()
                                if viewer is not None:
                                    viewer.sync()
                                count += 1
                                if count >= budget:
                                    return count
                                if viewer is not None:
                                    deadline += 1 / (args.fps * args.speed)
                                    time.sleep(max(0, deadline - time.monotonic()))
                    finally:
                        sim.free()
    if count == 0:
        raise ValueError("No nonempty episodes to replay")
    return count


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tasks = discover_tasks(args.data_dir, args.task or ("all" if args.list else None))
        if args.list:
            import h5py
            for task in tasks:
                with h5py.File(task, "r") as file:
                    print(task)
                    for name in episode_names(file["data"]):
                        states = file["data"][name].get("states")
                        print(f"  {name}: {len(states) if states is not None else 'missing'} states")
            return 0
        print(f"Restored {replay(args, tasks)} frames")
        return 0
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
