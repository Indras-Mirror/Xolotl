#!/usr/bin/env python3
"""Exercise the patched CUDA graph clock's actual metadata function without CUDA.

The generated C++ replaces only backend/graph storage and the monotonic clock.
No model, driver, or GPU is loaded. Keep generated files in --out for inspection.
"""
import argparse
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--llama-dir', type=Path, default=Path('third_party/llama.cpp'))
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--cxx', default='c++')
args = ap.parse_args()
source = (args.llama_dir / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
start = source.index('int64_t ggml_backend_cuda_graph_cache_shift_age(')
opening = source.index('{', start)
end, depth = opening + 1, 1
while depth:
    depth += (source[end] == '{') - (source[end] == '}')
    end += 1
function = source[start:end]
program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>
struct Graph { int64_t last_used_time = 0; std::vector<int> instances; };
struct ggml_backend_cuda_context {
    std::map<int, std::unique_ptr<Graph>> cuda_graphs;
    int64_t last_graph_eviction_sweep = 0;
};
struct Backend { bool cuda; void * context; };
using ggml_backend_t = Backend *;
bool ggml_backend_is_cuda(ggml_backend_t b) { return b && b->cuda; }
int64_t ggml_time_us() { return 1000; }
ACTUAL_FUNCTION
int main() {
    ggml_backend_cuda_context ctx;
    ctx.cuda_graphs[0] = std::make_unique<Graph>(Graph{100, {1,2,3}});
    ctx.cuda_graphs[1] = std::make_unique<Graph>(Graph{0, {4,5}});
    ctx.last_graph_eviction_sweep = 500;
    Backend cuda{true, &ctx}, other{false, &ctx};
    assert(ggml_backend_cuda_graph_cache_shift_age(nullptr, 200) == 0);
    assert(ggml_backend_cuda_graph_cache_shift_age(&other, 200) == 0);
#ifndef USE_CUDA_GRAPH
    assert(ggml_backend_cuda_graph_cache_shift_age(&cuda, 200) == 0);
    assert(ctx.cuda_graphs[0]->last_used_time == 100);
    assert(ctx.last_graph_eviction_sweep == 500);
#else
    assert(ggml_backend_cuda_graph_cache_shift_age(&cuda, 200) == 5);
    assert(ctx.cuda_graphs[0]->last_used_time == 300);
    assert(ctx.cuda_graphs[1]->last_used_time == 0); // sentinel remains uninitialized
    assert(ctx.last_graph_eviction_sweep == 700);
    assert(ggml_backend_cuda_graph_cache_shift_age(&cuda, -1) == 5);
    assert(ctx.cuda_graphs[0]->last_used_time == 300);
    assert(ggml_backend_cuda_graph_cache_shift_age(&cuda, 0) == 5);
    assert(ctx.last_graph_eviction_sweep == 700);
    ctx.cuda_graphs[1]->last_used_time = 5000;
    assert(ggml_backend_cuda_graph_cache_shift_age(&cuda, std::numeric_limits<int64_t>::max()) == 5);
    assert(ctx.cuda_graphs[0]->last_used_time == 1000);
    assert(ctx.cuda_graphs[1]->last_used_time == 1000);
    assert(ctx.last_graph_eviction_sweep == 1000);
    assert(ctx.cuda_graphs.size() == 2 && ctx.cuda_graphs[0]->instances.size() == 3);
#endif
}
'''.replace('ACTUAL_FUNCTION', function)
args.out.mkdir(parents=True, exist_ok=True)
cpp = args.out / 'checkpoint-clock.cpp'
cpp.write_text(program)
for graphs in (False, True):
    exe = args.out / ('checkpoint-clock-' + ('graphs' if graphs else 'no-graphs'))
    cmd = [args.cxx, '-std=c++17', '-O2', str(cpp), '-o', str(exe)]
    if graphs:
        cmd.append('-DUSE_CUDA_GRAPH')
    subprocess.run(cmd, check=True)
    subprocess.run([str(exe)], check=True, timeout=10)
print('checkpoint clock: graph and non-graph metadata checks PASS')
