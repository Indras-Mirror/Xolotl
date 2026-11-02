#ifdef NDEBUG
#undef NDEBUG
#endif
#include "strata/core/persistent_prefix.hpp"
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
    // Independent model processes share the same byte quota rather than multiplying it.
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
    const auto before=model_identity("test","settings",{model});
    {std::ofstream out(model); out << "456";}
    assert(before!=model_identity("test","settings",{model}));
    std::filesystem::remove_all(dir); // only this test's disposable fixtures
    std::cout << "persistent_prefix_test: PASS\n";
}
