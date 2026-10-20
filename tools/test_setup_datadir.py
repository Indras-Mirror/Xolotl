#!/usr/bin/env python3
"""Xolotl #6: Xolotl next to an original Strata install (same parent folder) gets its own settings and its own
Xolotl-data folder, never treats the Strata folder as one of its installs, and an Xolotl folder that ran before the
change keeps the data folder the shared Strata settings gave it.  No GPU, no downloads.

    python3 tools/test_setup_datadir.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load_setup(root: Path):
    spec = importlib.util.spec_from_file_location("xolotl_setup_under_test", HERE / "setup.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT = root
    m.WIN = False
    return m


def make_install(folder: Path, xolotl: bool):
    folder.mkdir(parents=True)
    (folder / "setup.py").write_text("# stub\n")
    if xolotl:
        (folder / "tools" / "v41").mkdir(parents=True)


class DataDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.portable, self.cfg = t / "portable", t / "config"
        self.strata, self.xolotl = self.portable / "Strata", self.portable / "Xolotl"
        make_install(self.strata, xolotl=False)
        make_install(self.xolotl, xolotl=True)
        (self.portable / "Strata-data" / "models").mkdir(parents=True)
        (self.portable / "Strata-data" / "models" / "m.gguf").write_text("x")
        self.env = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = str(self.cfg)

    def tearDown(self):
        if self.env is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self.env
        self.tmp.cleanup()

    def strata_settings(self, installs, data_dir):
        p = self.cfg / "strata" / "settings.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"installs": [str(i) for i in installs], "data_dir": str(data_dir)}))

    def test_fresh_xolotl_beside_strata(self):
        self.strata_settings([self.strata], self.portable / "Strata-data")
        m = load_setup(self.xolotl)
        self.assertEqual(m.settings_path(), self.cfg / "xolotl" / "settings.json")
        self.assertEqual(m.load_settings(), {})                       # Strata's settings are not ours
        self.assertNotIn(self.strata.resolve(), m.other_installs({}))  # the Strata folder is never an install
        dest, _ = m.data_folder(None)
        self.assertEqual(dest, self.portable / "Xolotl-data")
        self.assertTrue((self.portable / "Strata-data" / "models" / "m.gguf").exists())   # Strata's files untouched

    def test_existing_xolotl_keeps_its_data_folder(self):
        self.strata_settings([self.strata, self.xolotl], self.portable / "Strata-data")
        m = load_setup(self.xolotl)
        s = m.load_settings()
        self.assertEqual(s["data_dir"], str(self.portable / "Strata-data"))
        self.assertEqual([Path(p) for p in s["installs"]], [self.xolotl])   # the Strata folder is dropped
        dest, _ = m.data_folder(None)
        self.assertEqual(dest, self.portable / "Strata-data")

    def test_xolotl_siblings_still_found(self):
        make_install(self.portable / "Xolotl-main (1)", xolotl=True)
        m = load_setup(self.xolotl)
        self.assertIn((self.portable / "Xolotl-main (1)").resolve(), m.other_installs({}))


if __name__ == "__main__":
    unittest.main()
