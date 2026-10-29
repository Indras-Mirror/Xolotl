// Exercise the production transport/serve loop without model files or GPU work.
#define M31_PROTOCOL_TEST_ONLY
#include "../m31_generate.cpp"
namespace {
struct FixtureEngine {
    struct Geometry { int vocab = 8, embd = 4; } geometry_;
    int64_t position_ = 0;
    int last_n_ = 0;
    bool valid_ = true, bad_logits_ = false;
    const Geometry & geometry() const { return geometry_; }
    int64_t position() const { return position_; }
    int prefill_chunk(int requested, std::string &) { return std::min(64, requested); }
    bool forward(const int32_t * ids, int n, std::string &) {
        if (position_ == 0 && ids[0] == 6) bad_logits_ = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        position_ += n; last_n_ = n; return valid_;
    }
    bool forward_mixed(const int32_t * ids, const int32_t *, const float *, int, int n, std::string & error) {
        return forward(ids, n, error);
    }
    bool logits(int row, int count, std::vector<float> & result, std::string &) {
        if (row < 0 || row + count > last_n_) return false;
        result.assign(size_t(count * geometry_.vocab), -1);
        result[3] = 1;
        if (bad_logits_) result[0] = std::numeric_limits<float>::quiet_NaN();
        return true;
    }
    bool release_prefill_workspace(std::string &) { last_n_ = 0; return true; }
    bool rewind(int64_t position, std::string &) { position_ = position; last_n_ = 0; return valid_; }
    void reset() { position_ = 0; last_n_ = 0; valid_ = true; bad_logits_ = false; }
};
struct FixtureGrammar {
    bool active_ = false;
    int vocab_size() const { return 8; }
    std::set<int> stops() const { return {7}; }
    std::set<int> suppress() const { return {}; }
    void prepare(const std::string & encoded, const std::vector<int32_t> &, bool = false) {
        active_ = encoded == "allow2";
        if (encoded == "broken") throw std::runtime_error("fixture grammar rejected");
    }
    bool active() const { return active_; }
    bool allowed(int token) const { return !active_ || token == 2; }
    std::set<int> allowed_ids() const { return {2}; }
    void accept(int) {}
};
}
int main() {
    FixtureEngine engine;
    FixtureGrammar grammar;
    Config config;
    config.context = 4096;
    config.max_chunk = 64;
    Cli cli;
    Lines lines;
    bool quit = false, cancelled = false;
    config.cancelled = [&] { cancelled = cancelled || lines.stopped(quit); return cancelled; };
    serve(engine, config, cli, lines, quit, cancelled, grammar);
}
