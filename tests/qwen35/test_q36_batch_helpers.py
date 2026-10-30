"""Portable exact production-helper extraction/count/row-offset regression.

CMake generation: python test_q36_batch_helpers.py --generate-header OUTPUT
Standalone: python -m unittest discover -s tests/qwen35 -p test_q36_batch_helpers.py
CPU fakeK only; no CUDA or model inference/numerical correctness claim.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'src/qwen35/q36_batch_forward.hpp'

def extract():
    text=SOURCE.read_text()
    def method(name):
        start=text.index('void '+name+'(');brace=text.index('{',start);depth=0
        for index in range(brace,len(text)):
            if text[index]=='{':depth+=1
            if text[index]=='}':
                depth-=1
                if depth==0:return text[start:index+1]
        raise ValueError('unterminated production helper '+name)
    return '\n\n'.join(method(name) for name in ('batch_apply','batch_swiglu_quantize'))

def generate(path):
    methods=extract();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('// Exact methods extracted from src/qwen35/q36_batch_forward.hpp\n'+methods+'\n')
    return hashlib.sha256(methods.encode()).hexdigest()

class BatchHelpers(unittest.TestCase):
    def test_actual_helpers_tile_and_cover_every_row(self):
        supplied=os.environ.get('Q36_BATCH_HELPERS_MOCK_BIN')
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            if supplied:binary=Path(supplied).resolve()
            else:
                generate(tmp/'q36_batch_helpers_extracted.hpp')
                binary=tmp/'q36_batch_helpers_mock'
                compiler=shlex.split(os.environ.get('CXX','c++'))
                subprocess.run([*compiler,'-std=c++17','-O0','-I',str(tmp),str(HERE/'q36_batch_helpers_mock.cpp'),'-o',str(binary)],check=True)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS 36 cases',result.stdout)

if __name__=='__main__':
    if '--generate-header' in sys.argv:
        parser=argparse.ArgumentParser();parser.add_argument('--generate-header',required=True)
        args=parser.parse_args();generate(args.generate_header)
    else:unittest.main()
