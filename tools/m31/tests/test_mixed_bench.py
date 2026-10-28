#!/usr/bin/env python3
"""Reject partial, re-chunked or differently encoded mixed numeric evidence."""
from array import array
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from compare_mixed import check, PR_SHA
from prepare_vision import export_prepared


class MixedBench(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mixed = self.root / "mixed"
        self.mixed.mkdir()
        arrays = {"text_ids.i32": array("i", [1, -1]), "visual_rows.i32": array("i", [-1, 0]),
                  "positions.i32": array("i", [0, 1]), "visual.f32": array("f", [0.] * 6144)}
        for name, value in arrays.items():
            if sys.byteorder != "little":
                value.byteswap()
            (self.mixed / name).write_bytes(value.tobytes())
        (self.mixed / "manifest.json").write_text(json.dumps({"format": "m31-vision-batch-v1",
            "source_pr": PR_SHA, "encoded": True, "llm_inference": False, "n_embd": 6144,
            "rows": 2, "visual_rows": 1,
            "chunks": [{"index": 0, "row": 0, "position": 0, "n_pos": 1, "rows": 1,
                        "type": "text", "tokens": [1]},
                       {"index": 1, "row": 1, "position": 1, "n_pos": 1, "rows": 1,
                        "type": "image", "visual_row": 0, "decoder_positions": {"start": 1, "step": 1, "axes": 1}}]}))
        self.ref, self.cand = self.root / "ref", self.root / "cand"
        for path, engine in ((self.ref, "llama.cpp-PR31192"), (self.cand, "xolotl-m31")):
            path.mkdir()
            (path / "tokens.i32").write_bytes((self.mixed / "text_ids.i32").read_bytes())
            values = array("f", [0., 1., 2., 2., 1., 0.])
            if sys.byteorder != "little":
                values.byteswap()
            (path / "logits.f32").write_bytes(values.tobytes())
            (path / "manifest.json").write_text(json.dumps({"format": "m31-logits-v1", "kv": "f16",
                "model": "fixture.gguf", "tokens": 2, "vocab": 3, "positions": [0, 1], "backend": "cuda",
                "engine": engine, "oracle_op_offload": True, "context": 4096, "cuda_fusion_disabled": True,
                "mixed_manifest": str(self.mixed / "manifest.json"), "actual_chunks": [{"position": 0, "tokens": 2}]}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, **kwargs):
        return check(self.ref, self.cand, self.mixed / "manifest.json", **kwargs)

    def change(self, path, **values):
        data = json.loads(path.read_text())
        path.write_text(json.dumps(dict(data, **values)))

    def prepared_fixture(self):
        manifest = {"format": "m31-vision-ipc-v1", "source_pr": PR_SHA, "rows": 2, "visual_rows": 1,
            "n_embd": 6144, "encoded": True, "llm_inference": False,
            "chunks": [{"type": "text", "rows": 1, "tokens": [1]},
                       {"type": "image", "rows": 1, "embeddings": "visual.f32"}]}
        ipc = self.mixed / "ipc.json"
        ipc.write_text(json.dumps(manifest))
        sve = self.root / "prepared.sve"
        sve.write_bytes(struct.pack("<5i", 0x31455653, 1, 1, 1, 6144) +
                        (self.mixed / "visual.f32").read_bytes())
        return ipc, sve

    def test_export_uses_actual_prepared_sve_and_ids(self):
        ipc, sve = self.prepared_fixture()
        manifest = export_prepared([1, -1], sve, ipc, self.root / "export")
        self.assertEqual((manifest.parent / "visual.f32").read_bytes(),
                         (self.mixed / "visual.f32").read_bytes())

    def test_export_rejects_wrong_final_template_ids(self):
        ipc, sve = self.prepared_fixture()
        with self.assertRaisesRegex(ValueError, "Service.prepare IDs"):
            export_prepared([2, -1], sve, ipc, self.root / "export")

    def test_export_rejects_sve_not_produced_by_helper(self):
        ipc, sve = self.prepared_fixture()
        payload = bytearray(sve.read_bytes())
        payload[-1] = 128
        sve.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "SVE1 embeddings differ"):
            export_prepared([1, -1], sve, ipc, self.root / "export")

    def test_complete_rows_pass(self):
        self.assertTrue(self.run_check()["pass"])

    def test_identical_values_with_different_sign_zero_fail_raw_gate(self):
        data = bytearray((self.cand / "logits.f32").read_bytes())
        data[3] = 128  # +0 and -0 compare numerically equal, but differ in raw F32.
        (self.cand / "logits.f32").write_bytes(data)
        self.assertFalse(self.run_check()["pass"])

    def test_partial_rows_rejected(self):
        self.change(self.cand / "manifest.json", positions=[1])
        (self.cand / "logits.f32").write_bytes((self.cand / "logits.f32").read_bytes()[12:])
        with self.assertRaisesRegex(ValueError, "every prompt row"):
            self.run_check()

    def test_separate_frame_prefills_rejected(self):
        self.change(self.cand / "manifest.json", actual_chunks=[{"position": 0, "tokens": 1},
                                                              {"position": 1, "tokens": 1}])
        with self.assertRaisesRegex(ValueError, "one actual"):
            self.run_check()

    def test_fusion_or_context_mismatch_rejected(self):
        for change in ({"cuda_fusion_disabled": False}, {"context": 8192}):
            with self.subTest(change=change):
                before = (self.cand / "manifest.json").read_text()
                self.change(self.cand / "manifest.json", **change)
                with self.assertRaises(ValueError):
                    self.run_check()
                (self.cand / "manifest.json").write_text(before)

    def test_wrong_declared_visual_provenance_rejected(self):
        other = self.root / "other"
        other.mkdir()
        for path in self.mixed.iterdir():
            (other / path.name).write_bytes(path.read_bytes())
        data = bytearray((other / "visual.f32").read_bytes())
        data[0] = 1
        (other / "visual.f32").write_bytes(data)
        self.change(self.cand / "manifest.json", mixed_manifest=str(other / "manifest.json"))
        with self.assertRaisesRegex(ValueError, "dump declared mixed-input arrays differ"):
            self.run_check()

    def test_different_encoder_output_rejected(self):
        other = self.root / "independent"
        other.mkdir()
        for path in self.mixed.iterdir():
            (other / path.name).write_bytes(path.read_bytes())
        data = bytearray((other / "visual.f32").read_bytes())
        data[0] = 1
        (other / "visual.f32").write_bytes(data)
        with self.assertRaisesRegex(ValueError, "independently encoded"):
            self.run_check(independent_manifest=other / "manifest.json")


if __name__ == "__main__":
    unittest.main()
