"""Lightweight CLI / discovery tests; no simulator, torch, or network required."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_script("replay_libero")
downloader = load_script("download_libero")


class ReplayTests(unittest.TestCase):
    def test_defaults_and_episode_selection(self):
        args = replay.build_parser().parse_args([])
        self.assertEqual(args.data_dir, Path("datasets/libero_spatial"))
        self.assertEqual(args.episode, ["demo_0"])
        args = replay.build_parser().parse_args(["--episode", "2,demo_0,2", "--headless", "--max-frames", "3"])
        self.assertEqual(args.episode, ["demo_2", "demo_0"])
        self.assertTrue(args.headless)
        self.assertEqual(args.max_frames, 3)
        self.assertIsNone(replay.episode_selection("all"))

    def test_invalid_options(self):
        for option in (["--fps", "0"], ["--speed", "nan"], ["--max-frames", "0"], ["--episode", "-1"]):
            with self.subTest(option=option), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                replay.build_parser().parse_args(option)

    def test_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No .hdf5"):
                replay.discover_tasks(root)
            for name in ("b_bowl.hdf5", "a_bowl.hdf5", "ignored.txt"):
                (root / name).touch()
            self.assertEqual(replay.discover_tasks(root)[0].name, "a_bowl.hdf5")
            self.assertEqual(len(replay.discover_tasks(root, "all")), 2)
            self.assertEqual(replay.discover_tasks(root, "b_bowl")[0].name, "b_bowl.hdf5")
            self.assertEqual(replay.discover_tasks(root, str(root / "a_bowl.hdf5")), [(root / "a_bowl.hdf5").resolve()])
            with self.assertRaisesRegex(ValueError, "matched 2"):
                replay.discover_tasks(root, "bowl")

    def test_episode_order_and_missing(self):
        data = {"demo_10": None, "demo_2": None, "demo_0": None, "metadata": None}
        self.assertEqual(replay.episode_names(data), ["demo_0", "demo_2", "demo_10"])
        self.assertEqual(replay.episode_names(data, ["demo_2"]), ["demo_2"])
        with self.assertRaisesRegex(ValueError, "Episodes not found"):
            replay.episode_names(data, ["demo_7"])

    def test_isolated_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"LIBERO_CONFIG_PATH": "untouched-user-config"}):
                with replay.isolated_libero_config(root, root):
                    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
                    config = json.loads((config_dir / "config.yaml").read_text())
                    self.assertEqual(config["assets"], str(root / "assets"))
                self.assertFalse(config_dir.exists())
                self.assertEqual(os.environ["LIBERO_CONFIG_PATH"], "untouched-user-config")

    def test_asset_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rs = root / "installed_robosuite"
            libero = root / "installed_libero"
            for path in (rs / "models/assets/mesh.stl", libero / "assets/objects/texture.png"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            xml = '''<mujoco><asset>
              <mesh file="/old/robosuite/robosuite/models/assets/mesh.stl"/>
              <texture file="/old/LIBERO/libero/libero/assets/objects/texture.png"/>
              </asset><worldbody><body name="unchanged"/></worldbody></mujoco>'''
            tree = ET.fromstring(replay.repair_model_xml(xml.encode(), rs, libero))
            self.assertEqual(tree.find("asset/mesh").get("file"), str(rs / "models/assets/mesh.stl"))
            self.assertEqual(tree.find("asset/texture").get("file"), str(libero / "assets/objects/texture.png"))
            self.assertEqual(tree.find("worldbody/body").get("name"), "unchanged")
            legacy_xml = xml.replace("/old/LIBERO/libero/libero/", "/old/chiliocosm/")
            legacy = ET.fromstring(replay.repair_model_xml(legacy_xml, rs, libero))
            self.assertEqual(legacy.find("asset/texture").get("file"), str(libero / "assets/objects/texture.png"))
            with self.assertRaisesRegex(ValueError, "Model asset not found"):
                replay.repair_model_xml('<mujoco><asset><mesh file="/missing.stl"/></asset></mujoco>', rs, libero)


class Response(io.BytesIO):
    status = 200
    headers = {}


class DownloadTests(unittest.TestCase):
    def test_selection(self):
        entries = [{"path": "libero_spatial/a_bowl.hdf5"}, {"path": "libero_spatial/b_bowl.hdf5"}]
        self.assertEqual(downloader.select_tasks(entries, "all"), entries)
        self.assertEqual(downloader.select_tasks(entries, "A_BOWL"), entries[:1])
        with self.assertRaisesRegex(ValueError, "matched 2"):
            downloader.select_tasks(entries, "bowl")
        self.assertEqual(downloader.build_parser().parse_args([]).data_dir, Path("datasets/libero_spatial"))

    def test_download_resume_and_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "task.hdf5.part"
            partial.write_bytes(b"abc")
            response = Response(b"def")
            response.status = 206
            response.headers = {"Content-Range": "bytes 3-5/6"}
            entry = {"path": "libero_spatial/task.hdf5", "size": 6}
            with patch.object(downloader, "urlopen", return_value=response) as request:
                downloader.download(entry, root)
                self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
            self.assertEqual((root / "task.hdf5").read_bytes(), b"abcdef")
            self.assertFalse(partial.exists())
            with patch.object(downloader, "urlopen") as request:
                downloader.download(entry, root)
                request.assert_not_called()

    def test_range_ignored_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task.hdf5.part").write_bytes(b"bad")
            with patch.object(downloader, "urlopen", return_value=Response(b"complete")):
                downloader.download({"path": "libero_spatial/task.hdf5", "size": 8}, root)
            self.assertEqual((root / "task.hdf5").read_bytes(), b"complete")


if __name__ == "__main__":
    unittest.main()
