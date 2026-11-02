// Host-only compatibility identity for native DS41 persistence. No device API calls.
#pragma once
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/persistent_prefix.hpp"
#include <functional>
namespace strata::v41 {
// Conservative input identity: moving a model may miss, but changing file identity,
// CLI arithmetic settings or environment prevents reuse of an old image.
// As with session files, model inputs must remain immutable during a serving process.
inline std::string ds41_persistent_identity(int argc, char ** argv,
        const std::vector<std::string> & paths, const std::vector<std::string> & environment,
        const std::string & runtime) {
    if (runtime.empty()) return {};
    try {
    namespace fs = std::filesystem;
    std::ostringstream settings;
    settings << runtime.size() << ':' << runtime << ':';
    for (int i = 1; i < argc; ++i) settings << std::strlen(argv[i]) << ':' << argv[i] << ':';
    auto env = environment; std::sort(env.begin(), env.end());
    for (const auto & value : env) settings << value.size() << ':' << value << ':';
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto & input : paths) {
        if (input.empty()) continue;
        const fs::path p(input);
        if (fs::is_regular_file(p, ec) && !ec) {
            // Match GgufModel::open / Ds41FileBlobs / DSpark: ANY split shard
            // resolves to the entire ordered family, with missing shards refused.
            for (const auto & shard : strata::gguf_split_paths(p.string())) files.emplace_back(shard);
        }
        else if (fs::is_directory(p, ec) && !ec) {
            fs::recursive_directory_iterator it(p, ec), end;
            if (ec) return {};
            for (; it != end; it.increment(ec)) {
                if (ec) return {};
                if (it->is_regular_file(ec)) files.push_back(it->path());
                if (ec || files.size() > 65536) return {};
            }
        } else return {};
    }
    if (files.empty()) return {};
    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());
    std::vector<std::string> names;
    for (const auto & file : files) names.push_back(file.string());
    // Settings may contain private environment values; retain only a corruption-style
    // digest in metadata, never the original values or command line.
    uint64_t digest = 1469598103934665603ull;
    for (const unsigned char value : settings.str()) { digest ^= value; digest *= 1099511628211ull; }
    return strata::core::prefix::model_identity("ds41-native-boundary-v2", std::to_string(digest), std::move(names));
    } catch (...) { return {}; }
}

// The native adapter supplies a query of the already initialized backend. Keeping
// this gate here also makes the disabled-path/no-probe invariant testable on CPU.
inline std::string ds41_persistent_identity_if_enabled(bool enabled, int argc, char ** argv,
        const std::vector<std::string> & paths, const std::vector<std::string> & environment,
        const std::function<std::string()> & query_runtime) {
    if (!enabled || !query_runtime) return {};
    try { return ds41_persistent_identity(argc, argv, paths, environment, query_runtime()); }
    catch (...) { return {}; }
}
} // namespace strata::v41
