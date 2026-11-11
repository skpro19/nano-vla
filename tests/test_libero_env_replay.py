"""Action-replay unit tests using only stdlib fakes; no simulator required."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from test_libero_scripts import load_script


replay = load_script("replay_in_libero_env")


class Array:
    """Minimal dataset interface, deliberately not a numpy or h5py object."""

    def __init__(self, rows, shape=None):
        self.rows = rows
        self.shape = shape if shape is not None else (len(rows), len(rows[0]) if rows else 0)
        self.ndim = len(self.shape)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Group(dict):
    def __init__(self, items=(), **attrs):
        super().__init__(items)
        self.attrs = attrs


class ParserTests(unittest.TestCase):
    def test_defaults(self):
        args = replay.build_parser().parse_args([])
        self.assertEqual(args.data_dir, Path("datasets/libero_spatial"))
        self.assertEqual(args.episode, ["demo_0"])
        self.assertIsNone(args.fps)
        self.assertIsNone(args.max_frames)
        self.assertFalse(args.headless)
        self.assertFalse(args.compare_states)
        self.assertFalse(args.no_random)

    def test_compare_and_playback_options(self):
        args = replay.build_parser().parse_args([
            "--compare-states", "--headless", "--fps", "12", "--max-frames", "3",
        ])
        self.assertTrue(args.compare_states)
        self.assertTrue(args.headless)
        self.assertEqual(args.fps, 12)
        self.assertEqual(args.max_frames, 3)


class EnvironmentSpecTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bddl = self.root / "bddl_files/libero_spatial/task.bddl"
        self.bddl.parent.mkdir(parents=True)
        self.bddl.touch()
        self.controller = {"type": "OSC_POSE", "kp": 150, "impedance_mode": "fixed"}
        self.kwargs = {
            "controller_configs": self.controller,
            "control_freq": 20,
            "robots": ["Panda"],
            "bddl_file_name": "/obsolete/task.bddl",
            "has_renderer": True,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "ignore_done": False,
            "hard_reset": True,
        }

    def data(self, recorded="/old/LIBERO/bddl_files/libero_spatial/task.bddl", **kwargs):
        metadata = {
            "problem_name": "libero_tabletop_manipulation",
            "bddl_file": "/obsolete/bddl_files/missing.bddl",
            "env_kwargs": dict(self.kwargs, **kwargs),
        }
        return Group(env_args=json.dumps(metadata), bddl_file_name=recorded)

    def test_recorded_controller_and_top_level_bddl_take_precedence(self):
        data = self.data()
        problem, kwargs, frequency = replay.environment_spec(data, self.root)
        self.assertEqual(problem, "libero_tabletop_manipulation")
        self.assertEqual(kwargs["controller_configs"], self.controller)
        self.assertEqual(kwargs["robots"], ["Panda"])
        self.assertEqual(kwargs["bddl_file_name"], str(self.bddl.resolve()))
        self.assertEqual(frequency, 20.0)
        for key in ("has_renderer", "has_offscreen_renderer", "use_camera_obs", "hard_reset"):
            self.assertIs(kwargs[key], False, key)
        self.assertIs(kwargs["ignore_done"], True)
        # Overrides must not mutate the stored metadata.
        self.assertEqual(json.loads(data.attrs["env_args"])["env_kwargs"], self.kwargs)

    def test_windows_bytes_bddl_path(self):
        data = self.data(recorded=b"C:\\old\\bddl_files\\libero_spatial\\task.bddl")
        self.assertEqual(replay.environment_spec(data, self.root)[1]["bddl_file_name"],
                         str(self.bddl.resolve()))

    def test_metadata_bddl_fallback_and_default_frequency(self):
        data = self.data()
        del data.attrs["bddl_file_name"]
        metadata = json.loads(data.attrs["env_args"])
        metadata["bddl_file"] = "/old/bddl_files/libero_spatial/task.bddl"
        del metadata["env_kwargs"]["control_freq"]
        data.attrs["env_args"] = json.dumps(metadata)
        _, kwargs, frequency = replay.environment_spec(data, self.root)
        self.assertEqual(kwargs["bddl_file_name"], str(self.bddl.resolve()))
        self.assertEqual(frequency, 20.0)

    def test_missing_or_unrecognized_bddl(self):
        for recorded in ("", "/somewhere/task.bddl", "/old/bddl_files/missing.bddl"):
            with self.subTest(recorded=recorded), self.assertRaisesRegex(ValueError, "BDDL|bddl_files"):
                replay.environment_spec(self.data(recorded=recorded), self.root)

    def test_bddl_cannot_escape_root(self):
        outside = self.root / "outside.bddl"
        outside.touch()
        for recorded in ("/old/bddl_files/../outside.bddl", "/old/bddl_files/escape.bddl"):
            link = self.root / "bddl_files/escape.bddl"
            if not link.exists():
                link.symlink_to(outside)
            with self.subTest(recorded=recorded), self.assertRaisesRegex(ValueError, "Task BDDL not found"):
                replay.environment_spec(self.data(recorded=recorded), self.root)

    def test_invalid_recorded_frequency(self):
        for frequency in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(frequency=frequency), self.assertRaisesRegex(ValueError, "finite and positive"):
                replay.environment_spec(self.data(control_freq=frequency), self.root)


class InitializeEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/checkout/libero/libero")
        self.robosuite_root = Path("/robosuite")
        self.env = Mock()
        self.env.sim.get_state.return_value.flatten.return_value.size = 3
        self.states = Array([[1, 2, 3], [4, 5, 6]])
        self.episode = Group({"states": self.states}, model_file="recorded xml")
        repair_patch = patch.object(replay, "repair_model_xml", return_value="repaired xml")
        self.repair = repair_patch.start()
        self.addCleanup(repair_patch.stop)

    def initialize(self):
        replay.initialize_episode(self.env, self.episode, self.root, self.robosuite_root)

    def test_restores_only_initial_state_once_in_order(self):
        self.initialize()
        self.repair.assert_called_once_with("recorded xml", self.robosuite_root, self.root)
        self.assertEqual(self.env.mock_calls, [
            call.reset(), call.reset_from_xml_string("repaired xml"), call.sim.reset(),
            call.sim.get_state(), call.sim.get_state().flatten(),
            call.sim.set_state_from_flattened(self.states[0]), call.sim.forward(),
            call._update_observables(force=True),
        ])
        self.env.sim.set_state_from_flattened.assert_called_once_with(self.states[0])

    def test_rejects_invalid_state_shape_before_reset(self):
        for states in (Array([], shape=(0, 3)), Array([1, 2], shape=(2,)),
                       Array([[[1]]], shape=(1, 1, 1))):
            with self.subTest(shape=states.shape):
                self.episode["states"] = states
                with self.assertRaisesRegex(ValueError, "nonempty 2D"):
                    self.initialize()
        self.env.reset.assert_not_called()
        self.repair.assert_not_called()

    def test_rejects_state_width_before_setting_state(self):
        self.episode["states"] = Array([[1, 2]])
        with self.assertRaisesRegex(ValueError, "States have width 2; environment expects 3"):
            self.initialize()
        self.env.sim.set_state_from_flattened.assert_not_called()
        self.env.sim.forward.assert_not_called()
        self.env._update_observables.assert_not_called()

    def test_headless_replay_steps_recorded_actions_without_more_state_restores(self):
        actions = Array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self.episode["actions"] = actions
        data = Group({"demo_0": self.episode})
        self.env.action_dim = 2
        self.env._check_success.return_value = False
        factory = Mock(return_value=self.env)
        modules = {name: ModuleType(name) for name in (
            "h5py", "numpy", "robosuite", "libero", "libero.libero", "libero.libero.envs",
        )}
        modules["h5py"].File = Mock(side_effect=lambda *args: nullcontext({"data": data}))
        modules["numpy"].isfinite = Mock(return_value=SimpleNamespace(all=lambda: True))
        modules["robosuite"].__file__ = str(self.robosuite_root / "__init__.py")
        modules["libero.libero.envs"].TASK_MAPPING = {"test_problem": factory}
        args = replay.build_parser().parse_args(["--headless", "--no-random"])
        with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), \
                patch.object(replay, "libero_root", return_value=self.root), \
                patch.object(replay, "isolated_libero_config", return_value=nullcontext()), \
                patch.object(replay, "environment_spec", return_value=("test_problem", {}, 20)), \
                patch("sys.stdout", new=io.StringIO()):
            count = replay.replay(args, [Path("task.hdf5")])
        self.assertEqual(count, 3)
        self.assertEqual(self.env.step.call_args_list, [call(action) for action in actions])
        self.env.sim.set_state_from_flattened.assert_called_once_with(self.states[0])
        self.env.close.assert_called_once_with()
        factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
