#!/usr/bin/env python3
"""Small, CPU-only correctness gates for the Qwen4Exp edited XQ pipeline."""
import contextlib
import io
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v41"))
import apply_ternary as APPLY
import logits_gate as LG
import q8_ablate as Q
import tensor_backup as TB
from graft_ple import Hdr as TB_HDR
import v41_alloc as A
import v41_formats as F

sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
import gguf  # noqa: E402

MODEL = "/media/mal/NVME1TB/qwen38/qwen38-full-q8_0.gguf"

# The gate/up tiers the ablated chain actually screens (run_ablated_chain.sh).
# Kept here as a literal so a ladder change that forgets the screen fails a test.
SCREEN_TIERS = ["ptq1", "iq2_xxs", "iq2_xs", "iq2_s",
                "iq3_xxs", "iq3_s", "iq4_nl", "q8_0"]


class EditMath(unittest.TestCase):
    def test_full_rank_and_rank1(self):
        rng = np.random.default_rng(17)
        for rank in (1, 33):
            w = rng.standard_normal((2, 64, 256)).astype(np.float32)
            b = rng.standard_normal((2, 64, rank)).astype(np.float32)
            a = rng.standard_normal((2, rank, 256)).astype(np.float32)
            np.testing.assert_allclose(Q.apply_pair(w, b, a, 1.0, False), w + b @ a,
                                       rtol=2e-6, atol=2e-6)
        w = rng.standard_normal((3, 256, 640)).astype(np.float32)
        b = rng.standard_normal((3, 256, 1)).astype(np.float32)
        u = b[:, :, 0] / np.linalg.norm(b[:, :, 0], axis=1)[:, None]
        want = w - 1.5 * np.einsum("eo,eoi->eoi", u,
                                   np.einsum("eo,eoi->ei", u, w)[:, None, :])
        got = Q.apply_pair(w, b, np.zeros((3, 1, 640), np.float32), 1.5, True)
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-6)

    def test_survival_subtracts_base_quant_noise(self):
        rng = np.random.default_rng(9)
        w = rng.standard_normal((3, 128)).astype(np.float32)
        encoded = F.encode_q8_0(w)
        base = np.stack([F.decode(F.Q8_0, row, 128) for row in encoded])
        m = Q.edit_survival(w, w, base)
        self.assertAlmostEqual(m["alpha_eff"], 1.0, places=5)
        self.assertEqual(m["signal_pct"], 0.0)  # no intended edit -> no signal denominator


class EncoderAgreement(unittest.TestCase):
    """The screen measures one encoder; the writer must emit the same bytes."""

    def test_importance_survives_npz_roundtrip(self):
        rng = np.random.default_rng(5)
        w = rng.standard_normal((128, 256)).astype(np.float32)
        im = np.abs(rng.standard_normal(256)).astype(np.float32)
        want = F.encode_rtn(w, "ptq1", im)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "imp.npz"
            with open(p, "wb") as f:      # a handle, so savez does not append .npz
                np.savez(f, L3_gu_7=im)
            got = F.encode_rtn(w, "ptq1", np.load(p)["L3_gu_7"])
        self.assertEqual(want.tobytes(), got.tobytes())

    def test_encoder_rejects_bad_importance(self):
        rng = np.random.default_rng(6)
        w = rng.standard_normal((64, 256)).astype(np.float32)
        for bad in (np.ones(255, np.float32), -np.ones(256, np.float32),
                    np.full(256, np.nan, np.float32)):
            with self.assertRaises(ValueError):
                F.encode_rtn(w, "ptq1", bad)


class Rollback(unittest.TestCase):
    """A hard link is not a rollback: it shares the inode the edit writes into."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.t.name)
        self.model = self.d / "model.gguf"
        self.model.write_bytes(b"PRISTINE" * 1024)
        self.report = self.d / "edit.json"

    def tearDown(self):
        self.t.cleanup()

    def test_hardlink_is_rejected(self):
        link = self.d / "rollback.gguf"
        os.link(self.model, link)
        with self.assertRaisesRegex(SystemExit, "hard link"):
            Q.prepare(self.model, None, self.report, link)

    def test_identical_path_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "itself"):
            Q.prepare(self.model, None, self.report, self.model)

    def test_size_mismatch_is_rejected(self):
        other = self.d / "short.gguf"
        other.write_bytes(b"x")
        with self.assertRaisesRegex(SystemExit, "size"):
            Q.prepare(self.model, None, self.report, other)

    def test_independent_copy_is_accepted(self):
        copy = self.d / "pristine.gguf"
        copy.write_bytes(self.model.read_bytes())
        Q.prepare(self.model, None, self.report, copy)
        info = json.loads(self.report.read_text())
        self.assertEqual(info["rollback_kind"], "independent-copy")
        self.assertEqual(info["source_sha256"], Q.digest(self.model))

    def test_wrong_recorded_sha_is_rejected(self):
        copy = self.d / "pristine.gguf"
        copy.write_bytes(self.model.read_bytes())
        with self.assertRaisesRegex(SystemExit, "SHA-256"):
            Q.prepare(self.model, None, self.report, copy, "0" * 64)


def write_tiny_gguf(path, tensors):
    """tensors: [(name, dims, type_id, payload_bytes)] -> a minimal valid GGUF.

    Exists because type 143 is outside gguf-py's enum, so every reader/writer in
    this pipeline that touches a PTQ1_0 tensor must parse the header itself.
    """
    import struct
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0)
    off = 0
    infos, blobs = [], []
    for name, dims, tt, payload in tensors:
        n = name.encode()
        head += struct.pack("<Q", len(n)) + n
        head += struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        head += struct.pack("<IQ", tt, off)
        infos.append((name, dims, tt, off))
        blobs.append(payload)
        off += (len(payload) + 31) // 32 * 32
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + bytes(pad))
        for payload in blobs:
            f.write(payload + bytes((-len(payload)) % 32))
    return infos


class HeaderParse(unittest.TestCase):
    """A PTQ1_0 (type 143) tensor must be readable by our own header parser."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.t.name) / "tiny.gguf"

    def tearDown(self):
        self.t.cleanup()

    def test_type_143_is_parsed_and_sized(self):
        # 4 rows x 128 elements -> 1 PTQ1_0 block per row = 28 bytes each
        payload = bytes(range(28)) * 4
        write_tiny_gguf(self.p, [("blk.0.ffn_gate_exps.weight", (128, 4), 143, payload)])
        h = TB_HDR(str(self.p))
        self.assertEqual(len(h.infos), 1)
        _s, _e, name, tt, dims, off = h.infos[0]
        self.assertEqual((name, tt, tuple(dims), off), (name, 143, (128, 4), 0))
        self.assertEqual(APPLY.size_of(143, (128, 4)), len(payload))

    def test_gguf_py_cannot_read_it_which_is_why_we_do_not_use_it(self):
        write_tiny_gguf(self.p, [("blk.0.ffn_gate_exps.weight", (128, 4), 143, bytes(112))])
        with self.assertRaises(ValueError):
            gguf.GGUFReader(str(self.p))


class Ladder(unittest.TestCase):
    """The DP walks the tier list by index, so it must be in increasing bytes."""

    def test_qwen_ladders_are_monotone(self):
        for key in ("tiers", "dn_tiers", "pristine_dn_tiers"):
            t = A.ARCH["qwen4exp"][key]
            b = [A.ARCH["qwen4exp"]["bpw"][x] for x in t]
            self.assertEqual(b, sorted(b), f"{key} is not increasing: {list(zip(t, b))}")

    def test_every_ladder_tier_has_a_screening_name(self):
        # screen.py writes keys as egu_<tier>/edn_<tier>; the allocator reads exactly those.
        for t in A.ARCH["qwen4exp"]["tiers"]:
            self.assertIn(t, SCREEN_TIERS, f"gate/up tier {t} is never measured")

    def test_top_rung_can_be_parent_identical(self):
        # asked for: full Q8 available for the most sensitive gate/up layers
        self.assertEqual(A.ARCH["qwen4exp"]["tiers"][-1], "q8_0")
        self.assertIn("q8_0", A.ARCH["qwen4exp"]["dn_tiers"])


class AllocMainPath(unittest.TestCase):
    """main() must configure the module globals items_for() reads.

    It previously assigned TIERS/BPW as LOCALS, leaving the DeepSeek-V4 defaults in
    place, so a qwen4exp run died at the first expert with KeyError: 'iq2_xxs'.
    """

    def test_globals_are_set_for_a_qwen_run(self):
        saved = A.TIERS, A.BPW, A.DN_TIERS, A.GU, A.DN
        try:
            with tempfile.TemporaryDirectory() as d:
                w = pathlib.Path(d)
                (w / "stats").mkdir()
                src = pathlib.Path("/tmp/smoke_L16.json")
                if src.exists():
                    (w / "stats" / "L02.json").write_text(src.read_text())
                else:
                    self.skipTest("no smoke stats available")
                alloc = w / "alloc.json"
                r = subprocess.run(
                    [sys.executable, str(HERE.parent / "v41" / "v41_alloc.py"),
                     "--work", str(w), "--arch", "qwen4exp", "--per-proj",
                     "--edit-downs", "--layer-choice", "--budget-gib", "2.0",
                     "--unseen-floor", "iq2_xxs", "--out", str(alloc)],
                    capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
                self.assertNotIn("KeyError", r.stderr)
                d2 = json.loads(alloc.read_text())
                self.assertTrue(d2["layer_choice"])
                # the point is that the run completed with the qwen ladder configured;
                # which legal down tier it bought depends on the budget, so assert legality
                self.assertIn(d2["map"]["2"]["7"]["dn"], ("iq4_nl", "q8_0"))
                self.assertIn(d2["map"]["2"]["7"]["gu"], A.ARCH["qwen4exp"]["tiers"])
        finally:
            A.TIERS, A.BPW, A.DN_TIERS, A.GU, A.DN = saved

    def test_default_budget_needs_no_q2_0(self):
        # The default budget used to look up "q2_0" by name; it is not on the qwen ladder.
        A.TIERS = list(A.ARCH["qwen4exp"]["tiers"])
        A.BPW = dict(A.ARCH["qwen4exp"]["bpw"])
        A.DN_TIERS = list(A.ARCH["qwen4exp"]["dn_tiers"])
        A.GU, A.DN = 2 * 2560 * 640, 2560 * 640
        a = types.SimpleNamespace(arch="qwen4exp", edit_downs=True, per_proj=True,
                                  min_eval=8, edge_weight=2.0, unseen_floor="iq2_xxs")
        s = {"y_energy": 1.0, "n_eval": 100}
        for t in A.TIERS:
            s["egu_" + t] = 1.0
        for t in A.DN_TIERS:
            s["edn_" + t] = 1.0
        items = A.items_for({0: {"experts": {"0": s}}}, a)
        self.assertNotIn("q2_0", [c[0] for it in items for c in it[3]])
        floor = sum(it[3][0][2] for it in items)   # what main() now uses
        self.assertGreater(floor, 0)


class Emitter(unittest.TestCase):
    """llama-quantize has no PTQ1_0 type, so a ptq1 layer needs a writable placeholder."""

    def emit(self, *extra):
        d = pathlib.Path(self.t.name)
        alloc = d / "alloc.json"
        alloc.write_text(json.dumps({"layer_choice": True, "edit_downs": True, "map": {
            "0": {str(e): {"gu": "ptq1", "dn": "iq4_nl"} for e in range(3)},
            "1": {str(e): {"gu": "iq2_xxs", "dn": "q8_0"} for e in range(3)}}}))
        return subprocess.run([sys.executable, str(HERE / "emit_tensor_types.py"),
                               str(alloc), "--per-proj", *extra],
                              capture_output=True, text=True)

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.t.cleanup()

    def test_ptq1_layer_gets_a_writable_type(self):
        r = self.emit()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("^blk\\.0\\.ffn_(gate|up)_exps\\.weight$=Q2_0", r.stdout)
        self.assertNotIn("PTQ1_0", r.stdout)
        self.assertIn("^blk\\.1\\.ffn_down_exps\\.weight$=Q8_0", r.stdout)

    def test_ptq1_placeholder_cannot_be_ptq1(self):
        r = self.emit("--ptq1-placeholder", "PTQ1_0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no such type", r.stderr)

    def test_ngram_table_rule_precedes_the_catch_all(self):
        # first-match-wins: if the catch-all came first, the 51.2B-param n-gram table
        # would be re-expanded to the dense type and the size gate would blow.
        lines = self.emit().stdout.strip().splitlines()
        ple = [i for i, x in enumerate(lines) if "per_layer_token_embd" in x]
        catch = [i for i, x in enumerate(lines) if x.startswith("^blk\\.(?!")]
        self.assertEqual((len(ple), len(catch)), (1, 1))
        self.assertLess(ple[0], catch[0])

    def test_experts_match_a_rule_before_the_catch_all(self):
        # A routed-expert tensor must never fall through to the dense type: it would
        # quietly be stored at the backbone precision instead of its measured tier.
        lines = self.emit().stdout.strip().splitlines()
        catch = next(i for i, x in enumerate(lines) if x.startswith("^blk\\.(?!"))
        pats = [re.compile(x.rsplit("=", 1)[0]) for x in lines[:catch]]
        for name in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                     "blk.1.ffn_down_exps.weight"):
            self.assertTrue(any(p.search(name) for p in pats),
                            f"{name} falls through to the dense catch-all")


class RangeRollback(unittest.TestCase):
    """A range backup is the rollback for an in-place edit; prove it round-trips."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.t.name)
        self.model = self.d / "model.bin"
        rng = np.random.default_rng(3)
        self.data = rng.integers(0, 256, 1 << 20, dtype=np.uint8).tobytes()
        self.model.write_bytes(self.data)
        self.man = self.d / "man.json"
        self.man.write_text(json.dumps([
            {"name": "a", "offset": 0, "nbytes": 4096, "type": "X"},
            {"name": "b", "offset": 300000, "nbytes": 65000, "type": "X"}]))
        self.names = ["a", "b"]
        self.bk = self.d / "bk"
        TB.backup(self.model, self.man, self.bk, verify=False)

    def tearDown(self):
        self.t.cleanup()

    def test_prepare_records_range_rollback(self):
        rep = self.d / "edit.json"
        Q.prepare(self.model, None, rep, range_backup=self.bk, names=self.names)
        st = json.loads(rep.read_text())
        self.assertEqual(st["rollback_kind"], "range-backup")
        self.assertEqual(set(st["tensor_sha256"]), {"a", "b"})

    def test_prepare_refuses_a_tensor_the_backup_misses(self):
        with self.assertRaisesRegex(SystemExit, "does not cover"):
            Q.prepare(self.model, None, self.d / "e.json", range_backup=self.bk,
                      names=["a", "b", "c"])

    def test_prepare_refuses_a_model_that_already_moved(self):
        b = bytearray(self.data)
        b[300000] ^= 0xFF
        self.model.write_bytes(bytes(b))
        with self.assertRaisesRegex(SystemExit, "already differs"):
            Q.prepare(self.model, None, self.d / "e.json", range_backup=self.bk,
                      names=self.names)

    def test_prepare_refuses_another_model(self):
        other = self.d / "other.bin"
        other.write_bytes(self.data)
        with self.assertRaisesRegex(SystemExit, "not "):
            Q.prepare(other, None, self.d / "e.json", range_backup=self.bk, names=self.names)

    def test_double_edit_guard_sees_a_changed_tensor(self):
        # The guard compares live bytes with the recorded per-tensor hashes.
        # It must not read the adapter dict where it wants the tensor.
        live = {"a": types.SimpleNamespace(data_offset=0, n_bytes=4096),
                "b": types.SimpleNamespace(data_offset=300000, n_bytes=65000)}
        state = {"tensor_sha256": {n: Q.sha_bytes(self.data[r["offset"]:r["offset"] + r["nbytes"]])
                                   for n, r in
                                   (("a", {"offset": 0, "nbytes": 4096}),
                                    ("b", {"offset": 300000, "nbytes": 65000}))}}
        Q.check_targets_unchanged(self.model, state, live, {"a": {"b": 1}, "b": {"b": 1}})
        b = bytearray(self.data)
        b[300000] ^= 0xFF
        self.model.write_bytes(bytes(b))
        with self.assertRaisesRegex(SystemExit, "double edit"):
            Q.check_targets_unchanged(self.model, state, live, live)

    def test_restore_returns_the_original_bytes(self):
        b = bytearray(self.data)
        b[0:4096] = b"\x00" * 4096
        b[300000:365000] = b"\x11" * 65000
        self.model.write_bytes(bytes(b))
        TB.restore(self.model, self.man, self.bk, verify=True)
        self.assertEqual(self.model.read_bytes(), self.data)


@unittest.skipUnless(pathlib.Path(MODEL).exists(), "calibration model not present")
class BackupDecode(unittest.TestCase):
    """q8_rows over a backup payload must equal q8_rows over the live tensor."""

    def test_backup_view_decodes_like_the_live_tensor(self):
        # Feed a live Q8 tensor's OWN bytes through the backup-backed view: the two
        # decode routes must agree, whether or not the edit has run yet.
        tensors = [t for t in gguf.GGUFReader(MODEL).tensors
                   if t.tensor_type == gguf.GGMLQuantizationType.Q8_0]
        t = min(tensors, key=lambda x: int(x.n_bytes))
        with open(MODEL, "rb") as f:
            raw = Q.read_range(f, int(t.data_offset), int(t.n_bytes))
        view = Q._BackupTensor(t, raw)
        np.testing.assert_array_equal(Q.q8_rows(view), Q.q8_rows(t))
        self.assertEqual(view.data.shape, t.data.shape)


class Alloc(unittest.TestCase):
    def setUp(self):
        A.TIERS = list(A.ARCH["qwen4exp"]["tiers"])
        A.BPW = dict(A.ARCH["qwen4exp"]["bpw"])
        A.DN_TIERS = list(A.ARCH["qwen4exp"]["dn_tiers"])
        A.GU, A.DN = 2 * 2560 * 640, 2560 * 640
        A.EDGE_HI = 45
        self.a = types.SimpleNamespace(arch="qwen4exp", edit_downs=True,
                                       per_proj=True, min_eval=8, edge_weight=2.0,
                                       unseen_floor="iq2_xxs")

    def stat(self, **kw):
        s = {"y_energy": 1.0, "n_eval": 100, "e_q2_0": 1.0}
        for t in A.TIERS:
            s["egu_" + t] = 0.1 if t == "ptq1" else 1.0
            s["e_" + t] = 1.0
        for t in A.DN_TIERS:
            s["edn_" + t] = 0.1 if t == "iq4_nl" else 0.2
        s.update(kw)
        return s

    def test_protected_down_and_ternary_gu(self):
        items = A.items_for({0: {"experts": {"0": self.stat()}}}, self.a)
        self.assertEqual([x[0] for x in items[0][3]], ["ptq1"] + A.TIERS[1:])
        self.assertEqual([x[0] for x in items[1][3]], ["iq4_nl", "q8_0"])
    def test_layer_knapsack_preserves_tier_intent(self):
        self.a.layer_choice = True
        stats = {0: {"experts": {str(e): self.stat() for e in range(4)}}}
        amap, used, err = A.solve_layers(stats, self.a, 4 * (A.GU * A.BPW["iq4_nl"] + A.DN * A.BPW["iq4_nl"]) / 8)
        choice = amap["0"]["0"]
        self.assertEqual(choice, {"gu": "ptq1", "dn": "iq4_nl"})
        self.assertEqual(len(amap["0"]), 4)
        self.assertLessEqual(used, 4 * (A.GU * A.BPW["iq4_nl"] + A.DN * A.BPW["iq4_nl"]) / 8)

    def test_knapsack_distinguishes_neighbouring_tiers(self):
        # A per-expert byte grid made PTQ1 and IQ2_XXS the same cost, so the DP could
        # never buy the cheaper error. At a 1 MiB grid it must.
        self.a.layer_choice = True
        s = self.stat(egu_ptq1=0.20, egu_iq2_xxs=0.05)
        stats = {0: {"experts": {str(e): dict(s) for e in range(4)}}}
        need = 4 * (A.GU * A.BPW["iq2_xxs"] + A.DN * A.BPW["iq4_nl"]) / 8
        amap, used, _ = A.solve_layers(stats, self.a, need + (8 << 20))
        self.assertEqual(amap["0"]["0"]["gu"], "iq2_xxs")
        self.assertLessEqual(used, need + (8 << 20))

    def test_knapsack_never_exceeds_budget(self):
        self.a.layer_choice = True
        stats = {0: {"experts": {str(e): self.stat() for e in range(4)}},
                 1: {"experts": {str(e): self.stat() for e in range(4)}}}
        floor = 8 * (A.GU * A.BPW["ptq1"] + A.DN * A.BPW["iq4_nl"]) / 8
        budget = floor + (16 << 20)
        amap, used, _ = A.solve_layers(stats, self.a, budget)
        self.assertLessEqual(used, budget)
        self.assertEqual(len(amap), 2)

    def test_missing_measurement_is_not_zero(self):
        s = self.stat()
        del s["egu_ptq1"]
        with self.assertRaisesRegex(ValueError, "missing measurement"):
            A.items_for({0: {"experts": {"0": s}}}, self.a)

    def test_measured_expert_without_combined_key_is_not_unseen(self):
        # An edited-down ladder writes no e_q2_0. Judging "unseen" by that key would
        # floor every expert and drop ternary gate/up from the file.
        s = {"y_energy": 1.0, "n_eval": 100}
        for t in A.TIERS:
            s["egu_" + t] = 0.1 if t == "ptq1" else 1.0
        for t in A.DN_TIERS:
            s["edn_" + t] = 0.1
        self.assertNotIn("e_q2_0", s)
        items = A.items_for({0: {"experts": {"0": s}}}, self.a)
        gu = next(it for it in items if it[2] == "gu")
        self.assertFalse(gu[4])
        self.assertEqual(gu[3][0][0], "ptq1")


class LogitsGate(unittest.TestCase):
    """strata's dump has a 2 x int32 header; reading it as headerless shifts every row by 8 bytes.
    These gates make that class of misread loud instead of authoritative."""

    VOCAB, ROWS = 12, 5

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        rng = np.random.default_rng(5)
        self.a = rng.standard_normal((self.ROWS, self.VOCAB)).astype(np.float32) * 4

    def dump(self, name, arr, header=True):
        p = self.tmp / name
        with open(p, "wb") as f:
            if header:
                f.write(struct.pack("<2i", arr.shape[1], arr.shape[0]))
            f.write(arr.astype(np.float32).tobytes())
        return str(p)

    def test_identical_dump_passes_with_zero_kl(self):
        pa, pb = self.dump("a.f32", self.a), self.dump("b.f32", self.a)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = LG.main([pa, pb, "--min-top1", "1.0", "--max-kl", "1e-12"])
        self.assertEqual(rc, 0)
        m = LG.compare(pa, pb)
        self.assertEqual(m["top1_agreement"], 1.0)
        self.assertLess(m["kl_mean"], 1e-12)
        self.assertEqual(m["max_abs_logit_diff"], 0.0)
        self.assertEqual((m["vocab"], m["rows"]), (self.VOCAB, self.ROWS))

    def test_truncated_dump_is_refused_not_reshaped(self):
        p = self.dump("t.f32", self.a)
        with open(p, "r+b") as f:
            f.truncate(os.path.getsize(p) - 4)
        with self.assertRaisesRegex(LG.DumpError, "truncated"):
            LG.load(p)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(LG.main([p, p]), 2)

    def test_perturbed_rows_are_caught_and_gate_fails(self):
        b = self.a.copy()
        b[::2] = b[::2][:, ::-1]                     # reverses each perturbed row: a different argmax
        pa, pb = self.dump("a.f32", self.a), self.dump("b.f32", b)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(LG.main([pa, pb, "--min-top1", "0.99"]), 1)
        m = LG.compare(pa, pb)
        self.assertLess(m["top1_agreement"], 1.0)
        self.assertGreater(m["kl_mean"], 0.0)
        self.assertGreater(m["max_abs_logit_diff"], 0.0)

    def test_headerless_needs_vocab_and_is_not_inferred(self):
        p = self.dump("h.f32", self.a, header=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(LG.main([p, p]), 2)                # header bytes are real logits: refuse
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(LG.main([p, p, "--vocab", str(self.VOCAB)]), 0)

    def test_vocab_mismatch_between_dumps_is_refused(self):
        other = self.a[:, :6].copy()
        pa, pb = self.dump("a.f32", self.a), self.dump("v6.f32", other)
        with self.assertRaisesRegex(LG.DumpError, "vocab mismatch"):
            LG.compare(pa, pb)

    def test_ppl_uses_the_row_that_predicts_the_next_token(self):
        ids = np.zeros(self.ROWS + 1, dtype=np.int32)
        ids_path = str(self.tmp / "ids.i32")
        ids.tofile(ids_path)
        pa = self.dump("a.f32", self.a)
        m = LG.compare(pa, pa, ids_path=ids_path)
        self.assertEqual(m["nll_tokens"], self.ROWS)   # row t predicts ids[t+1], so all ROWS rows count
        self.assertAlmostEqual(m["ppl_a"], m["ppl_b"], places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
