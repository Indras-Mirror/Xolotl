#include "m31_engine.hpp"
#include "m31_run.hpp"
#include "m31_protocol.hpp"
#include "m31_grammar.hpp"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <cerrno>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

namespace {
using namespace strata::m31;
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
struct Cli {
    bool serve = false;
    int device = 0;
    std::string stop, suppress;
    run::Args args;
    Config config;
};
[[maybe_unused]] Cli parse_cli(int argc, char ** argv) {
    Cli cli;
    std::vector<std::string> filtered{argv[0]};
    std::string serve_model;
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--serve") cli.serve = true;
    if (cli.serve) {
        cli.args.chunk = cli.config.max_chunk;
        cli.args.threads = cli.config.threads;
#ifdef M31_ENGINE_CUDA
        cli.args.backend = "cuda";
#endif
    }
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--serve") continue;
        if (key == "--help" || key == "-h") throw std::runtime_error(
            "usage: m31_generate --serve -m MODEL [--backend cpu|cuda --device N --ctx N --chunk N --threads N "
            "--arena-gib N --gpu-experts-gib N --register-arena 0|1 --register-limit-gib N --stop id,id]\n"
            "offline: m31_generate -m MODEL (--ids-file x.i32|--cases DIR) --out NEW-DIR [--rows all|last|last:N|stride:N]");
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[++i];
        auto pos = [&](int high = std::numeric_limits<int>::max()) { return int(protocol::integer(value, 1, high, key.c_str())); };
        if (key == "--device") cli.device = int(protocol::integer(value, 0, 1023, "device"));
        else if (key == "--stop") cli.stop = value;
        else if (key == "--suppress") cli.suppress = value;
        else if (key == "--register-arena") cli.config.register_arena = protocol::integer(value, 0, 1, "register-arena") != 0;
        else if (key == "--register-limit-gib") cli.config.register_limit_bytes = uint64_t(protocol::real(value, 0, 65536, "register-limit-gib") * double(uint64_t(1) << 30));
        else if (key == "--register-chunk-mib") cli.config.register_chunk_bytes = uint64_t(pos()) << 20;
        else if (key == "--register-threads") cli.config.register_threads = unsigned(pos(64));
        else if (key == "--gpu-pack-threads") cli.config.gpu_pack_threads = unsigned(pos(4096));
        else if (key == "--upload-depth") cli.config.upload_depth = unsigned(pos(128));
        else if (key == "--upload-streams") cli.config.upload_streams = unsigned(pos(128));
        else if (key == "--prefetch-limit") cli.config.prefetch_limit = unsigned(pos(128));
        else if (key == "--cpu-fused") cli.config.cpu_fused = protocol::integer(value, 0, 1, "cpu-fused") != 0;
        else if (key == "--cpu-poll") cli.config.cpu_poll = int(protocol::integer(value, -1, 100, "cpu-poll"));
        else if (key == "--cpu-affinity") cli.config.cpu_affinity = value;
        else if (!cli.serve) { filtered.push_back(key); filtered.push_back(value); }
        else if (key == "-m" || key == "--model") cli.args.model = value;
        else if (key == "--backend") cli.args.backend = value;
        else if (key == "--ctx") cli.args.context = pos(1048576);
        else if (key == "--chunk" || key == "--prefill-chunk") cli.args.chunk = pos(32768);
        else if (key == "--threads") cli.args.threads = pos(4096);
        else if (key == "--arena-gib") cli.args.arena_gib = protocol::real(value, 0, 65536, "arena-gib");
        else if (key == "--gpu-experts-gib") cli.args.gpu_experts_gib = protocol::real(value, 0, 1024, "gpu-experts-gib");
        else throw std::runtime_error("unsupported serve option: " + key);
    }
    if (!cli.serve) {
        std::vector<char *> pointers;
        for (auto & arg : filtered) pointers.push_back(arg.data());
        cli.args = run::parse(int(pointers.size()), pointers.data());
    }
    const auto & a = cli.args;
    if (a.model.empty()) throw std::runtime_error("--model is required");
    if (a.backend != "cpu" && a.backend != "cuda") throw std::runtime_error("backend must be cpu or cuda");
    if (a.oracle_op_offload) throw std::runtime_error("--oracle-op-offload is oracle-only");
    auto & cfg = cli.config;
    cfg.threads = a.threads; cfg.context = a.context; cfg.max_chunk = a.chunk;
    cfg.arena_gib = a.arena_gib; cfg.gpu_experts_gib = a.gpu_experts_gib;
    cfg.gpu_experts_min_tokens = a.gpu_experts_min_tokens;
    cfg.reuse_allocations = a.reuse_allocations; cfg.persistent_cpu_pool = a.persistent_cpu_pool;
    cfg.cpu_residual = a.cpu_residual; cfg.cpu_decode_cache = a.cpu_decode_cache;
    cfg.dense_decode_cache = a.dense_decode_cache; cfg.profile = a.profile;
    return cli;
}
// The input thread uses bounded OS reads and wakes every 100 ms. QUIT can
// join it while the parent still holds stdin open; no detached std::cin reader
// can block process shutdown while holding the standard stream lock.
class Lines {
    struct State {
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<std::string> queue;
        bool eof = false, closed = false;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    std::thread reader_;
public:
    Lines() {
        reader_ = std::thread([state = state_] {
            auto enqueue = [&](std::string line) {
                std::unique_lock<std::mutex> lock(state->mutex);
                state->changed.wait(lock, [&] { return state->queue.size() < 64 || state->closed; });
                if (state->closed) return false;
                state->queue.push_back(std::move(line));
                state->changed.notify_all();
                return true;
            };
            std::string pending;
            bool oversized = false, finished = false;
            while (!finished) {
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    if (state->closed) break;
                }
                char buffer[4096];
                int count = 0;
#ifdef _WIN32
                const HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
                if (GetFileType(handle) == FILE_TYPE_PIPE) {
                    DWORD available = 0;
                    if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr)) break;
                    if (!available) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
                } else if (GetFileType(handle) == FILE_TYPE_CHAR && WaitForSingleObject(handle, 100) != WAIT_OBJECT_0) continue;
                DWORD received = 0;
                if (!ReadFile(handle, buffer, sizeof(buffer), &received, nullptr)) break;
                count = int(received);
#else
                pollfd descriptor{STDIN_FILENO, POLLIN, 0};
                const int ready = poll(&descriptor, 1, 100);
                if (ready < 0) { if (errno == EINTR) continue; break; }
                if (!ready) continue;
                const auto received = read(STDIN_FILENO, buffer, sizeof(buffer));
                if (received < 0) { if (errno == EINTR) continue; break; }
                count = int(received);
#endif
                if (!count) break;
                for (int i = 0; i < count; ++i) {
                    if (buffer[i] == '\n') {
                        if (!pending.empty() && pending.back() == '\r') pending.pop_back();
                        if (!enqueue(oversized ? "INVALID oversized command" : std::move(pending))) { finished = true; break; }
                        pending.clear(); oversized = false;
                    } else if (!oversized) {
                        if (pending.size() == 16 * 1024 * 1024) { pending.clear(); oversized = true; }
                        else pending += buffer[i];
                    }
                }
            }
            if (!pending.empty() || oversized) enqueue(oversized ? "INVALID oversized command" : std::move(pending));
            std::lock_guard<std::mutex> lock(state->mutex);
            state->eof = true;
            state->changed.notify_all();
        });
    }
    ~Lines() {
        { std::lock_guard<std::mutex> lock(state_->mutex); state_->closed = true; state_->changed.notify_all(); }
        if (reader_.joinable()) reader_.join();
    }
    bool next(std::string & line) {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->changed.wait(lock, [&] { return !state_->queue.empty() || state_->eof; });
        if (state_->queue.empty()) return false;
        line = std::move(state_->queue.front()); state_->queue.pop_front(); state_->changed.notify_all();
        return true;
    }
    bool stopped(bool & quit) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        // --serve is a live controller connection. Once stdin closes, no
        // controller can acknowledge progress or cancel a long request later.
        // Offline ID/logit runs do not construct this reader.
        bool stopped = state_->eof;
        if (state_->eof) quit = true;
        for (auto it = state_->queue.begin(); it != state_->queue.end();) {
            if (*it == "STOP" || it->rfind("BSTOP ", 0) == 0 || *it == "QUIT") {
                if (*it == "QUIT") quit = true;
                stopped = true; it = state_->queue.erase(it);
            } else ++it;
        }
        state_->changed.notify_all();
        return stopped;
    }
};
void error_line(const std::string & text) {
    std::string clean = text;
    std::replace(clean.begin(), clean.end(), '\n', ' ');
    std::replace(clean.begin(), clean.end(), '\r', ' ');
    std::cout << "ERR " << clean << '\n' << std::flush;
}
void done(int generated, int prompt, double prompt_ms, double decode_ms, const char * finish, int reused, int read, int slot) {
    std::cout << "DONE " << generated << ' ' << prompt << ' ' << prompt_ms << ' ' << decode_ms << ' ' << finish
              << " 0 0 " << reused << " 0 0 0 0 0 " << read << " 0\n";
    // A batch admission runs to completion on the serialized text worker. No
    // concurrent slots are advertised; BADM=0 tells Strata there is no tail slot.
    if (slot >= 0) std::cout << "BADM " << slot << " 0\n";
    std::cout << std::flush;
}
template<class EngineType, class GrammarType>
void serve(EngineType & engine, Config & config, const Cli & cli, Lines & lines, bool & quit, bool & cancelled, GrammarType & grammar) {
    if (grammar.vocab_size() && grammar.vocab_size() != engine.geometry().vocab) throw std::runtime_error("grammar vocabulary size mismatch");
    auto stops = grammar.stops(), suppress = grammar.suppress();
    const auto vocab = int(engine.geometry().vocab);
    if (!cli.stop.empty()) for (auto id : protocol::token_ids(cli.stop, vocab)) stops.insert(id);
    if (!cli.suppress.empty()) for (auto id : protocol::token_ids(cli.suppress, vocab)) suppress.insert(id);
    std::cout << "INFO prefix_cache=1 vision=0 logit_bias=1 batch=0 grammar="
#ifdef M31_HAVE_GRAMMAR
              << 1
#else
              << 0
#endif
              << "\nREADY " << config.context << " stop\n" << std::flush;
    std::vector<int32_t> history;
    std::string line;
    while (!quit && lines.next(line)) {
        if (line == "QUIT") break;
        if (line.empty() || line == "STOP" || line.rfind("BSTOP ", 0) == 0) continue;
        protocol::Request request;
        try {
            request = protocol::parse_request(line, vocab, config.context);
            grammar.prepare(request.chat_b64, request.ids);
        } catch (const std::exception & e) { error_line(e.what()); continue; }
        cancelled = false;
        // A sampler belongs to one generation, as in the accepted worker. An
        // unseeded request must not inherit another request's RNG sequence.
        std::mt19937_64 random(request.seeded ? request.seed : uint64_t(std::random_device{}()));
        std::string error;
        int reused = request.cache_prompt ? protocol::reusable_prefix(history, request.ids) : 0;
        if (reused && !engine.rewind(reused, error)) { history.clear(); engine.reset(); reused = 0; }
        else if (!reused) { engine.reset(); history.clear(); }
        if (reused) { history.resize(size_t(reused)); std::cout << "RESUME " << reused << '\n' << std::flush; }
        const auto tp = Clock::now();
        int read = 0, generated = 0, last_chunk = 0;
        double prompt_ms = 0, decode_ms = 0;
        Clock::time_point td;
        bool decoding = false;
        const char * finish = "length";
        try {
            for (int position = reused; position < int(request.ids.size());) {
                if (config.cancelled && config.cancelled()) { cancelled = true; break; }
                const int n = engine.prefill_chunk(std::min(config.max_chunk, int(request.ids.size()) - position), error);
                if (n <= 0 || !engine.forward(request.ids.data() + position, n, error)) {
                    if (cancelled) break;
                    throw std::runtime_error(error);
                }
                position += n; read += n; last_chunk = n;
                history.insert(history.end(), request.ids.begin() + position - n, request.ids.begin() + position);
                const double ms = elapsed_ms(tp);
                std::cout << "PP " << position << ' ' << request.ids.size() << ' ' << ms << ' '
                          << (1000.0 * read / std::max(ms, 1e-9)) << '\n' << std::flush;
            }
            if (!cancelled && config.cancelled && config.cancelled()) cancelled = true;
            std::vector<float> logits;
            if (!cancelled && request.max_new > 0) {
                // Keep the output-head graph's original last-chunk width until
                // its row is copied. Releasing scratch first changes its recipe.
                if (!engine.logits(last_chunk - 1, 1, logits, error))
                    throw std::runtime_error(error);
                if (!engine.release_prefill_workspace(error)) throw std::runtime_error(error);
            }
            for (; !cancelled && generated < request.max_new;) {
                if (config.cancelled && config.cancelled()) { cancelled = true; break; }
                auto sample_request = request;
                auto excluded = suppress;
                if (request.ignore_eos) excluded.insert(stops.begin(), stops.end());
                int token = protocol::sample_token(logits, sample_request, history, excluded, random);
                // Match the previous constrained sampler: test the sampled token
                // first, and only materialize the full grammar mask on rejection.
                if (grammar.active() && !grammar.allowed(token)) {
                    auto mask = grammar.allowed_ids();
                    if (sample_request.masked) {
                        std::set<int> intersection;
                        std::set_intersection(mask.begin(), mask.end(), sample_request.allowed.begin(), sample_request.allowed.end(),
                                              std::inserter(intersection, intersection.end()));
                        mask.swap(intersection);
                    }
                    sample_request.masked = true; sample_request.allowed = std::move(mask);
                    token = protocol::sample_token(logits, sample_request, history, excluded, random);
                }
                grammar.accept(token);
                const auto sampled_at = Clock::now();
                if (!decoding) {
                    // The prompt includes its head and first sample, as in the
                    // accepted worker and llama.cpp. Decode covers n-1 steps.
                    prompt_ms = std::chrono::duration<double, std::milli>(sampled_at - tp).count();
                    td = sampled_at;
                    decoding = true;
                } else {
                    decode_ms = std::chrono::duration<double, std::milli>(sampled_at - td).count();
                }
                std::cout << "T " << token << '\n' << std::flush;
                ++generated;
                if ((!request.ignore_eos && stops.count(token)) || request.stop.count(token)) { finish = "stop"; break; }
                if (generated == request.max_new) break;
                if (!engine.forward(&token, 1, error)) {
                    if (cancelled) break;
                    throw std::runtime_error(error);
                }
                history.push_back(token);
                if (!engine.logits(0, 1, logits, error)) throw std::runtime_error(error);
            }
            if (!decoding) prompt_ms = elapsed_ms(tp);
            if (cancelled) {
                finish = "cancel";
                if (decoding) decode_ms = elapsed_ms(td);
            }
            if (cancelled || !request.cache_prompt || !request.checkpoint) { engine.reset(); history.clear(); }
            done(generated, int(request.ids.size()), prompt_ms, decode_ms, finish, reused, read, request.slot);
        } catch (const std::exception & e) {
            engine.reset(); history.clear();
            if (cancelled) {
                if (decoding) decode_ms = elapsed_ms(td);
                else prompt_ms = elapsed_ms(tp);
                done(generated, int(request.ids.size()), prompt_ms, decode_ms, "cancel", reused, read, request.slot);
            } else if (std::string(e.what()) == "nonfinite logits") {
                if (decoding) decode_ms = elapsed_ms(td);
                else prompt_ms = elapsed_ms(tp);
                done(generated, int(request.ids.size()), prompt_ms, decode_ms, "nonfinite", reused, read, request.slot);
            } else error_line(e.what());
        }
    }
}
#ifndef M31_PROTOCOL_TEST_ONLY
void offline(Engine & engine, const run::Args & args, run::Dump * & active_dump) {
    if (!run::fs::create_directory(args.out)) throw std::runtime_error("output directory exists; use a new path");
    for (const auto & path : run::cases(args)) {
        const auto ids = run::read_ids(path);
        if (ids.size() > size_t(args.context)) throw std::runtime_error("case exceeds configured context");
        engine.reset();
        run::Dump dump(run::fs::path(args.out) / path.stem(), args, ids, int(engine.geometry().vocab));
        active_dump = &dump;
        for (int pos = 0; pos < int(ids.size());) {
            std::string error;
            const int n = engine.prefill_chunk(run::chunk_size(args, int(ids.size()), pos), error);
            if (n <= 0 || !engine.forward(ids.data() + pos, n, error)) throw std::runtime_error(error);
            dump.record_chunk(pos, n);
            const int count = args.head_tail ? std::min(n, args.head_tail) : n;
            const int first = n - count;
            std::vector<float> logits;
            if (!engine.logits(first, count, logits, error)) throw std::runtime_error(error);
            for (int row = first; row < n; ++row) if (dump.wants(pos + row))
                dump.row(pos + row, logits.data() + size_t(row - first) * engine.geometry().vocab);
            pos += n;
        }
        dump.finish("xolotl-m31", int(ids.size()));
        active_dump = nullptr;
    }
}
#endif
} // namespace

#ifndef M31_PROTOCOL_TEST_ONLY
int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
        std::cout << "m31_generate --serve -m MODEL [--backend cpu|cuda --device N --ctx N --chunk N --threads N]\n"
                     "             [--arena-gib N --gpu-experts-gib N --register-arena 0|1 --register-limit-gib N]\n"
                     "m31_generate -m MODEL (--ids-file CASE.i32|--cases DIR) --out NEW-DIR\n"
                     "             [--rows all|last|last:N|stride:N --head-tail N --decode-tail N]\n";
        return 0;
    }
    ggml_backend_t backend = nullptr;
    try {
        auto cli = parse_cli(argc, argv);
        // This executable owns its environment. The HTTP process and every other
        // model keep their original ggml settings.
        prepare_engine_environment();
        validate_config(cli.config, cli.args.backend == "cuda");
        if (cli.args.backend == "cuda") {
#ifdef M31_ENGINE_CUDA
            backend = ggml_backend_cuda_init(cli.device);
            if (!backend) throw std::runtime_error("cannot initialize requested CUDA device");
#else
            throw std::runtime_error("built without CUDA");
#endif
        }
        {
            Lines * input = nullptr;
            std::unique_ptr<Lines> lines;
            bool quit = false, cancelled = false;
            if (cli.serve) {
                lines = std::make_unique<Lines>(); input = lines.get();
                cli.config.cancelled = [&] { cancelled = cancelled || input->stopped(quit); return cancelled; };
            }
            cli.config.backend = backend;
            validate_config(cli.config);
            std::cerr << "m31 effective configuration: " << effective_config(cli.config) << '\n';
            run::Dump * active_dump = nullptr;
            if (!cli.serve && cli.args.taps != "none") cli.config.tap = [&](const auto & name, int64_t pos, const auto * tensor) {
                if (active_dump) active_dump->tensor(name, pos, tensor);
            };
            Engine engine;
            std::string error;
            if (!engine.init(cli.args.model, cli.config, error)) throw std::runtime_error(error);
            if (cli.serve) {
                ChatGrammar grammar(cli.args.model);
                serve(engine, cli.config, cli, *lines, quit, cancelled, grammar);
            }
            else offline(engine, cli.args, active_dump);
        }
        if (backend) ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "m31_generate: " << e.what() << '\n';
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}

#endif // M31_PROTOCOL_TEST_ONLY
