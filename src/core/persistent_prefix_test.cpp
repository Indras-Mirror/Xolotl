#ifdef NDEBUG
#undef NDEBUG
#endif
#include "strata/core/persistent_prefix.hpp"
#include "../qwen35/q36_persistent_payload.hpp"
#include <cassert>
#include <iostream>
#include <sys/wait.h>
using namespace strata::core::prefix;
int main() {
    char dir[] = "/tmp/strata-prefix-test-XXXXXX";
    assert(mkdtemp(dir));
    Config c; c.directory = dir; c.disk_bytes = 2400; c.min_free_bytes = 1; c.queue_bytes = 4096;
    const std::vector<uint8_t> payload(1000, 42);
    {
        Store s(c); assert(s.enabled());
        assert(s.enqueue("a", {1,2}, payload)); s.flush();
        assert(s.stats().writes == 1);
        assert(!s.find("b",{1,2,3}));
        auto r = s.find("a",{1,2,3}); assert(r && r->cells.size()==2 && r->payload==payload);
        assert(!s.find("a",{1,3}));
        assert(!s.find("a",{1,2},1));
        assert(!s.find("a",{1,2},SIZE_MAX,[]{return true;}));
        assert(!s.enqueue("a",{1},std::vector<uint8_t>(5000)));
    }
    // A genuine new process discovers only metadata and lazily reads the chosen payload.
    {
        Config d=c; d.directory=std::string(dir)+"/latency";
        d.min_restore_cells=2; d.nonblocking_lookup=true;
        Store s(d); assert(s.enabled());
        assert(s.enqueue("model",{1,2},payload));
        assert(s.enqueue("image",{0},payload)); s.flush();
        assert(!s.find("model",{1}));
        assert(!s.find("model",{1,2,3},SIZE_MAX,{},1)); // RAM already supplies all but one cell.
        assert(s.find("model",{1,2,3}));
        assert(!s.find("image",{0}));
        assert(s.find("image",{0},1,{},0,true)); // Encoded images are not token prefixes.
        int fd=::open((d.directory+"/lock").c_str(),O_RDWR|O_CLOEXEC);
        assert(fd>=0 && ::flock(fd,LOCK_EX|LOCK_NB)==0);
        assert(!s.find("model",{1,2,3})); // A busy writer is an immediate optional-cache miss.
        assert(::flock(fd,LOCK_UN)==0); ::close(fd);
        assert(s.find("model",{1,2,3})); // Contention never invalidates the saved record.
    }
    auto child = fork(); assert(child >= 0);
    if (!child) { Store s(c); auto r=s.find("a",{1,2,4}); _exit(r && r->payload==payload ? 0:1); }
    int status=0; assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    {
        Store s(c);
        assert(s.enqueue("b",{3,4},payload)); s.flush();
        assert(s.find("a",{1,2,3})); // Touch a, making b the quota victim.
        assert(s.enqueue("c",{5,6},payload)); s.flush();
        assert(s.find("a",{1,2})); assert(!s.find("b",{3,4})); assert(s.find("c",{5,6}));
        auto lease=s.find_file("a",{1,2}); assert(lease);
        // A separate process cannot acquire an evicting writer lease while the reader owns it.
        child=fork(); assert(child>=0);
        if(!child) { Store other(c); _exit(0); }
        usleep(10000); assert(waitpid(child,&status,WNOHANG)==0);
        lease.reset(); assert(waitpid(child,&status,0)==child && WIFEXITED(status));
        auto corrupt=s.find_file("a",{1,2}); assert(corrupt); const auto path=corrupt->path; corrupt.reset();
        { std::ofstream out(path,std::ios::binary|std::ios::trunc); out << "broken"; }
        assert(!s.find("a",{1,2}));
        // Writer exceptions are misses, never a published catalogue entry.
        assert(s.enqueue_file("bad",{9},10,[](const std::string&,std::string&)->bool {throw 1;}));
        s.flush(); assert(!s.find("bad",{9})); assert(s.stats().failures==1);
    }
    // Capture claims suppress the expensive producer before copying payload bytes.
    {
        Config d=c; d.directory=std::string(dir)+"/claims"; d.queue_bytes=16384;
        Store s(d);
        auto first=s.claim_capture("model",{1,2}); assert(first);
        assert(!s.claim_capture("model",{1,2})); // pending capture
        first.reset();
        first=s.claim_capture("model",{1,2}); assert(first); // abandoned capture retries
        auto reservation=s.reserve(payload.size()); assert(reservation);
        assert(s.enqueue_reserved("model",{1,2},payload,std::move(reservation),std::move(first)));
        s.flush(); assert(s.stats().writes==1);
        assert(!s.claim_capture("model",{1,2})); // committed duplicate, no producer call
        assert(s.find("model",{1,2,3}));
        assert(!s.claim_capture("model",{1,2})); // LRU touch retains suppression
        auto different=s.claim_capture("other-model",{1,2}); assert(different);
        auto tail=s.claim_capture("model",{1,3}); assert(tail);
        auto shorter=s.claim_capture("model",{1}); assert(shorter); // retain every anchor
        auto file=s.find_file("model",{1,2}); assert(file); auto path=file->path; file.reset();
        {std::ofstream out(path,std::ios::binary|std::ios::trunc);
         const std::vector<uint8_t> corrupt(payload.size(),11);
         out.write(reinterpret_cast<const char*>(corrupt.data()),corrupt.size());}
        assert(!s.find("model",{1,2}));
        auto retry=s.claim_capture("model",{1,2}); assert(retry); // corruption retries
        reservation=s.reserve(payload.size()); assert(reservation);
        assert(s.enqueue_reserved("model",{1,2},payload,std::move(reservation),std::move(retry)));
        s.flush(); assert(!s.claim_capture("model",{1,2}));
        assert(s.enqueue("evict-a",{7},payload)); s.flush();
        assert(s.enqueue("evict-b",{8},payload)); s.flush();
        assert(!s.find("model",{1,2}));
        assert(s.claim_capture("model",{1,2})); // externally evictable, never permanently seen
        auto failed=s.claim_capture("failed",{9}); assert(failed);
        auto held_failed=failed;
        assert(s.enqueue_file_keys_reserved("failed",{{9}},10,
            [](const std::string&,std::string&){return false;},{},std::move(failed)));
        s.flush(); assert(s.claim_capture("failed",{9})); // writer failure retries
        auto dropped=s.claim_capture("drop",{10}); assert(dropped);
        auto held_dropped=dropped;
        assert(!s.enqueue_file_keys_reserved("drop",{{10}},5000,
            [](const std::string&,std::string&){return true;},{},std::move(dropped)));
        assert(s.claim_capture("drop",{10})); // admission failure retries
    }
    {
        Config d=c; d.directory=std::string(dir)+"/claim-bounds"; d.max_entries=1; d.queue_bytes=16384;
        Store s(d);
        auto pending=s.claim_capture("one",{1}); assert(pending);
        auto uncached=s.claim_capture("two",{2}); assert(uncached); // cache full still captures
        assert(!s.claim_capture("one",{1}));
        auto reservation=s.reserve(payload.size()); assert(reservation);
        assert(s.enqueue_reserved("two",{2},payload,std::move(reservation),std::move(uncached)));
        s.flush(); assert(s.find("two",{2}));
    }
    {
        Config d=c; d.directory=std::string(dir)+"/claim-race"; d.queue_bytes=16384;
        Store s(d);
        std::atomic<int> ready{0}, attempted{0}, winners{0}; std::atomic<bool> start{false};
        std::vector<std::thread> producers;
        for(int i=0;i<8;++i) producers.emplace_back([&] {
            ++ready; while(!start.load())std::this_thread::yield();
            auto claim=s.claim_capture("same",{1,2,3}); if(claim)++winners;
            ++attempted; while(attempted.load()!=8)std::this_thread::yield();
        });
        while(ready.load()!=8)std::this_thread::yield(); start=true;
        for(auto& producer:producers)producer.join();
        assert(winners==1); assert(s.claim_capture("same",{1,2,3}));
        // Even untracked claim metadata consumes the queue budget before host capture.
        Config b=d;b.directory=std::string(dir)+"/claim-pressure";b.max_entries=1;b.disk_bytes=65536;
        Store bounded(b); auto first=bounded.claim_capture("first",{1});assert(first);
        auto pressure=bounded.reserve(b.queue_bytes-2048);assert(pressure);
        auto untracked=bounded.claim_capture("second",std::vector<uint64_t>(1024,2));assert(!untracked);
        pressure.reset();untracked=bounded.claim_capture("second",std::vector<uint64_t>(1024,2));assert(untracked);
        assert(!bounded.reserve(b.queue_bytes-1024));
    }
    // Independent model processes share the same byte quota rather than multiplying it.
    // Cancel inside the real checksum/payload loops, without timing or slow disks.
    // A successful lookup calibrates the deterministic callback sequence for this
    // isolated one-entry directory; the final calls belong to 64 KiB hash chunks.
    {
        Config d=c; d.directory=std::string(dir)+"/cancel-read";
        d.disk_bytes=8ull<<20; d.queue_bytes=8ull<<20;
        Store s(d); const std::vector<uint8_t> large(3ull<<20, 17);
        assert(s.enqueue("cancel",{1,2},large)); s.flush();
        size_t lookup_polls=0;
        auto file=s.find_file("cancel",{1,2},SIZE_MAX,[&]{++lookup_polls;return false;});
        assert(file && lookup_polls>16); file.reset();
        const auto hits=s.stats().hits.load();
        size_t polls=0;
        assert(!s.find_file("cancel",{1,2},SIZE_MAX,[&]{return ++polls==lookup_polls-8;}));
        assert(polls==lookup_polls-8 && s.stats().hits==hits);
        // find() first repeats the lookup, then polls before each 1 MiB read.
        // Cancel before the second payload chunk, after one actual chunk read.
        polls=0;
        assert(!s.find("cancel",{1,2},SIZE_MAX,[&]{return ++polls==lookup_polls+2;}));
        assert(polls==lookup_polls+2);
        // Aborted readers release both the directory lease and payload budget.
        auto budget=s.reserve(d.queue_bytes); assert(budget); budget.reset();
        auto retry=s.find("cancel",{1,2}); assert(retry && retry->payload==large);
        retry.reset();
        // The same capture primitive used by Q36 must release admission state
        // when cancellation discards its partially copied private bytes.
        auto cancelled_capture=[&] {
            auto claim=s.claim_capture("partial",{3,4}); assert(claim);
            auto reservation=s.reserve(large.size()); assert(reservation);
            strata::q36::persistent::Writer writer;size_t checks=0;
            return writer.capture(large.data(),large.size(),[&]{return ++checks==2;});
        };
        assert(!cancelled_capture());
        assert(s.claim_capture("partial",{3,4}));
        budget=s.reserve(d.queue_bytes); assert(budget); budget.reset();
        assert(!s.find("partial",{3,4}));
    }
    std::vector<pid_t> writers;
    for(int i=0;i<4;++i) {
        child=fork(); assert(child>=0);
        if(!child) { Store s(c); assert(s.enqueue("writer"+std::to_string(i),{uint64_t(i+20)},payload));
                     s.flush(); _exit(s.stats().failures ? 1:0); }
        writers.push_back(child);
    }
    for(auto pid:writers) assert(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0);
    uint64_t used=0;
    for(const auto& item:std::filesystem::directory_iterator(dir))
        if(item.path().extension()==".bin" || item.path().extension()==".idx") used+=item.file_size();
    assert(used<=c.disk_bytes);
    {
        Store s(c); auto hold=s.reserve(2300); assert(hold); assert(!s.reserve(2000));
        assert(s.enqueue_reserved("reserved",{77},payload,hold));
        assert(!s.enqueue_reserved("reserved-twice",{88},payload,hold));
        hold.reset(); s.flush(); assert(s.find("reserved",{77}));
    }
    {
        Config full=c; full.directory=std::string(dir)+"/full"; full.min_free_bytes=UINT64_MAX;
        Store s(full); assert(s.enabled()); assert(s.enqueue("full",{1},payload)); s.flush();
        assert(s.stats().failures==1 && !s.find("full",{1}));
    }
    { Config bad=c; ++bad.disk_bytes; Store s(bad); assert(!s.enabled()); }
    // Identity invalidates same-size writes even when mtime is restored.
    const auto model=std::string(dir)+"/model";
    {std::ofstream out(model); out << "123";}
    // Unbound embedding producers may use text persistence, never keyed image state.
    const char* inherited=std::getenv("STRATA_PREFIX_CACHE_VISION_ID");
    const std::optional<std::string> saved_vision=inherited?std::optional<std::string>(inherited):std::nullopt;
    unsetenv("STRATA_PREFIX_CACHE_VISION_ID");
    assert(image_state_allowed(false) && !image_state_allowed(true));
    const auto unbound=model_identity("test","settings",{model});
    const std::string binding="frontend-vision-pipeline-v1:"+std::string(64,'a');
    setenv("STRATA_PREFIX_CACHE_VISION_ID",binding.c_str(),1);
    assert(image_state_allowed(false) && image_state_allowed(true));
    const auto bound=model_identity("test","settings",{model});assert(bound!=unbound);
    setenv("STRATA_PREFIX_CACHE_VISION_ID",("frontend-vision-pipeline-v1:"+std::string(64,'b')).c_str(),1);
    assert(model_identity("test","settings",{model})!=bound);
    setenv("STRATA_PREFIX_CACHE_VISION_ID","stale/unvalidated",1);
    assert(image_state_allowed(false) && !image_state_allowed(true));
    if(saved_vision)setenv("STRATA_PREFIX_CACHE_VISION_ID",saved_vision->c_str(),1);
    else unsetenv("STRATA_PREFIX_CACHE_VISION_ID");
    const auto before=model_identity("test","settings",{model});
    {std::ofstream out(model); out << "456";}
    assert(before!=model_identity("test","settings",{model}));
    std::filesystem::remove_all(dir); // only this test's disposable fixtures
    std::cout << "persistent_prefix_test: PASS\n";
}
