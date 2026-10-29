#!/usr/bin/env python3
"""Model-free synthetic head comparator regressions; run only after quality/EX."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m31_cache_compare_test',HERE.parent/'bench'/'compare_prefix_heads.py')
C = importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


class StrictHeads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='m31-cache-head-test-')
        self.a,self.b = Path(self.tmp.name)/'A',Path(self.tmp.name)/'B'
        for path in (self.a,self.b): self.write_run(path)

    def tearDown(self): self.tmp.cleanup()

    def write_run(self,path,length=1124):
        path.mkdir()
        heads={}
        for name,recipe in C.expected_recipes(length).items():
            group = 0 if name in ('whole','whole_again') else 1 if name in ('whole_rewind','whole_decode_rewind') else 2
            values=[0.0]*8;values[group]=1.0
            (path/(name+'.f32')).write_bytes(struct.pack('<8f',*values))
            heads[name]={'top1':group,'forward_chunks':recipe}
        (path/'summary.json').write_text(json.dumps({'same_history_bitwise':True,'length':length,'reused':1024,'vocab':8,
            'whole_vs_split_bitwise':False,'whole_vs_replay_bitwise':False,'heads':heads}))

    def summary(self): return json.loads((self.a/'summary.json').read_text())
    def put_summary(self,data): (self.a/'summary.json').write_text(json.dumps(data))

    def test_production_vocabulary_fixed(self): self.assertEqual(C.VOCAB,200064)

    def test_all_heads_and_distinct_histories_pass(self):
        result=C.compare(self.a,self.b,1124,vocab=8)
        self.assertTrue(result['pass']);self.assertEqual(len(result['heads']),9)
        self.assertFalse(result['whole_vs_split_bitwise']['A'])

    def test_numeric_equal_signed_zero_is_not_byte_equal(self):
        for name in ('whole','whole_again'):
            values=[1.0]+[0.0]*6+[-0.0]
            (self.a/(name+'.f32')).write_bytes(struct.pack('<8f',*values))
        result=C.compare(self.a,self.b,1124,vocab=8)
        self.assertFalse(result['pass']);self.assertEqual(result['heads']['whole']['changed_vocabulary_entries'],1)

    def test_missing_extra_and_short_head_rejected(self):
        file=self.a/'restore.f32';raw=file.read_bytes();file.unlink()
        with self.assertRaisesRegex(ValueError,'missing or extra'):C.load_run(self.a,1124,vocab=8)
        file.write_bytes(raw);extra=self.a/'other.f32';extra.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'missing or extra'):C.load_run(self.a,1124,vocab=8)
        extra.unlink();file.write_bytes(raw[:-4])
        with self.assertRaisesRegex(ValueError,'full-vocabulary'):C.load_run(self.a,1124,vocab=8)

    def test_changed_recipe_and_boolean_count_rejected(self):
        data=self.summary();data['heads']['split']['forward_chunks'][1]['tokens']=99;self.put_summary(data)
        with self.assertRaisesRegex(ValueError,'recipe'):C.load_run(self.a,1124,vocab=8)
        data['heads']['split']['forward_chunks']=C.expected_recipes(1124)['split']
        data['heads']['whole_decode_rewind']['forward_chunks'][0]['tokens']=True;self.put_summary(data)
        with self.assertRaisesRegex(ValueError,'recipe'):C.load_run(self.a,1124,vocab=8)

    def test_finite_top1_and_invariant_rejected(self):
        file=self.a/'restore.f32';raw=file.read_bytes();values=list(struct.unpack('<8f',raw));values[7]=float('nan')
        file.write_bytes(struct.pack('<8f',*values))
        with self.assertRaisesRegex(ValueError,'non-finite'):C.load_run(self.a,1124,vocab=8)
        file.write_bytes(raw);data=self.summary();data['heads']['restore']['top1']=0;self.put_summary(data)
        with self.assertRaisesRegex(ValueError,'top1'):C.load_run(self.a,1124,vocab=8)
        data['heads']['restore']['top1']=2;self.put_summary(data);values[7]=.25
        file.write_bytes(struct.pack('<8f',*values))
        with self.assertRaisesRegex(ValueError,'invariant'):C.load_run(self.a,1124,vocab=8)

    def test_all_300_suffix_recipes_and_decode_rows(self):
        a,b=Path(self.tmp.name)/'A300',Path(self.tmp.name)/'B300'
        self.write_run(a,1324);self.write_run(b,1324)
        self.assertTrue(C.compare(a,b,1324,vocab=8)['pass'])
        recipe=C.expected_recipes(1324)['split_decode_rewind']
        self.assertEqual(recipe[-1],{'position':1024,'tokens':300})
        self.assertEqual(recipe[:-1],[{'position':1324+i,'tokens':1} for i in range(7)])


if __name__=='__main__':unittest.main(verbosity=2)
