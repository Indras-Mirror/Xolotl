"""Execute the native env gate and input routing with no GPU dependencies."""
from pathlib import Path
import os
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[1]
source = (repo / 'src/program/generate.cpp').read_text()
a = source.index('        const bool batch_cancel_enabled =')
b = source.index('        const auto cancel_mailbox =', a)
gate = source[a:b]
a = source.index('                strata::program::BatchStop stop;', source.index('while (getline_fd(l))'))
b = source.index('                std::lock_guard<std::mutex> lk(st->mu);', a)
route = source[a:b]
a = source.index('batch_cancel_enabled ? (cancel_fd >= 0 ? "batch_cancel_epoch=1')
b = source.index(',\n', a)
prefix = source[a:b]
program = r'''
#include "strata/program/batch_cancel.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
struct Mail { int requests=0; void request(int,uint64_t){++requests;} };
int main() {
'''
program += 'for (const char* value : {"", "0", "1", "true"}) { setenv("STRATA_BATCH_CANCEL_PIPE",value,1);\n' + gate
program += 'assert(batch_cancel_enabled == (std::string(value)=="1"));\n'
program += 'for(int cancel_fd : {-1, 9}) {const std::string p = ' + prefix + ';\n'
program += 'assert((!batch_cancel_enabled && p.empty()) || (batch_cancel_enabled && p.find("batch_cancel_epoch=1 ")==0));}\n'
program += 'auto mail=std::make_shared<Mail>();int queued=0;for(const std::string l : {"BSTOP 0 1","BSTOP 0","BSTOP 0 malformed"}) {\n' + route
program += '++queued;}assert(mail->requests==(batch_cancel_enabled?1:0));assert(queued==(batch_cancel_enabled?1:3));}\n}\n'
with tempfile.TemporaryDirectory(prefix='strata-cancel-protocol-') as temp:
    cpp = Path(temp) / 'protocol.cpp'
    exe = Path(temp) / 'protocol'
    cpp.write_text(program)
    subprocess.run([os.environ.get('CXX','c++'), '-std=c++20', '-I'+str(repo/'include'), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print('PASS actual native env/INFO prefix/input routing: OFF unchanged, ON tagged, legacy fallback')
