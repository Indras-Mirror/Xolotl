// Per-step training metrics for the loss graphs (tools/train/lossplot.py reads RUNS/<run>/metrics.jsonl).
//   xolotl::train::Metrics m("runs", "qfn-harness-r16", config_json);
//   m.log(step, {{"loss", l}, {"lr", lr}, {"gnorm", g}, {"tokens", (double) tok}});   // "t" (seconds) is added
// One JSON object per line, flushed every line, so the page can be regenerated while the run is going.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>   // _mkdir: MSVC's sys/stat.h declares no POSIX mkdir
#define mkdir(path, mode) _mkdir(path)
#endif
#include <utility>
#include <vector>

namespace xolotl::train {

struct Metrics {
    std::FILE * f = nullptr;
    std::string dir;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Metrics(const std::string & root, const std::string & run, const std::string & config_json) {
        ::mkdir(root.c_str(), 0775);
        dir = root + "/" + run;
        ::mkdir(dir.c_str(), 0775);
        if (std::FILE * c = std::fopen((dir + "/config.json").c_str(), "w")) {
            std::fputs(config_json.c_str(), c);
            std::fclose(c);
        }
        f = std::fopen((dir + "/metrics.jsonl").c_str(), "a");
    }
    ~Metrics() { if (f) std::fclose(f); }
    void log(long step, const std::vector<std::pair<const char *, double>> & kv) {
        if (!f) return;
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(f, "{\"step\": %ld, \"t\": %.2f", step, t);
        for (const auto & p : kv)
            if (std::isfinite(p.second)) std::fprintf(f, ", \"%s\": %.6g", p.first, p.second);
        std::fputs("}\n", f);
        std::fflush(f);
    }
};

}  // namespace xolotl::train
