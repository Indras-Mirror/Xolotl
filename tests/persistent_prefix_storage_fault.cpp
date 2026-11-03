#ifdef NDEBUG
#undef NDEBUG
#endif
#include "strata/core/persistent_prefix.hpp"
#include <cassert>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <poll.h>
#include <sys/statfs.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::core::prefix;
namespace fs = std::filesystem;
constexpr size_t MiB=1024*1024;
// Real filesystem writes, not a synthetic error return or a Store implementation copy.
static bool write_bytes(int fd,size_t bytes,int& error) {
    const std::vector<unsigned char> block(64*1024,0x53);
    while(bytes) {
        auto n=::write(fd,block.data(),std::min(bytes,block.size()));
        if(n<0 && errno==EINTR) continue;
        if(n<=0) {error=n<0?errno:EIO;return false;}
        bytes-=size_t(n);
    }
    return true;
}
static size_t files_with(const fs::path& dir,const char* suffix) {
    size_t n=0;for(const auto& p:fs::directory_iterator(dir))n+=p.path().extension()==suffix;return n;
}
static Config config(const fs::path& dir) {
    Config c;c.directory=dir.string();c.disk_bytes=32*MiB;c.queue_bytes=8*MiB;c.min_free_bytes=1;return c;
}
int main() {
    const char* root=std::getenv("STRATA_PREFIX_FAULT_DIR");
    if(!root || !*root) {std::cout<<"SKIP: STRATA_PREFIX_FAULT_DIR requires a dedicated small tmpfs\n";return 77;}
    assert(geteuid()!=0);struct statfs info{};assert(::statfs(root,&info)==0);
    assert(info.f_type==0x01021994); // TMPFS_MAGIC: never fill an ordinary/shared data filesystem.
    assert(uint64_t(info.f_blocks)*info.f_bsize<=16*MiB && uint64_t(info.f_bavail)*info.f_bsize>=7*MiB);
    const auto run=fs::path(root)/("prefix-fault-"+std::to_string(getpid()));assert(fs::create_directory(run));
    const auto pressure=run/"owned-pressure.bin";
    std::atomic<int> observed_errno{0};std::atomic<bool> writer_entered{false};
    {
        auto c=config(run/"enospc");Store store(c);assert(store.enabled());
        assert(store.enqueue_file("fault",{1,2},4*MiB,[&](const std::string& target,std::string&)->bool {
            writer_entered=true; // Store has already passed its real free-space/quota admission.
            int error=0;int filler=::open(pressure.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);assert(filler>=0);
            assert(write_bytes(filler,6*MiB,error));assert(::close(filler)==0);
            int fd=::open(target.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);assert(fd>=0);
            bool ok=write_bytes(fd,4*MiB,error);if(!ok)observed_errno=error;::close(fd);return ok;
        }));
        store.flush();assert(writer_entered && observed_errno==ENOSPC);
        assert(store.stats().failures==1 && store.stats().writes==0);
        assert(!store.find("fault",{1,2}));assert(files_with(c.directory,".idx")==0);
        assert(files_with(c.directory,".part")==0 && files_with(c.directory,".bin")==0);
        auto full=store.reserve(c.queue_bytes);assert(full);full.reset(); // All queued/private bytes released.
        // Free only this fixture's own pressure allocation; retain the file and all receipts.
        assert(::truncate(pressure.c_str(),0)==0);
        assert(store.enqueue("recovered",{3,4},std::vector<uint8_t>(4096,42)));store.flush();
        auto result=store.find("recovered",{3,4});assert(result && result->payload==std::vector<uint8_t>(4096,42));
    }
    {
        Store fresh(config(run/"enospc"));assert(!fresh.find("fault",{1,2}));assert(fresh.find("recovered",{3,4}));
    }
    auto c=config(run/"interrupted");
    {Store baseline(c);assert(baseline.enqueue("baseline",{8},std::vector<uint8_t>(4096,8)));baseline.flush();}
    int ready[2],gate[2];assert(pipe(ready)==0 && pipe(gate)==0);
    const pid_t parent_pid=getpid();
    pid_t child=fork();assert(child>=0);
    if(child==0) {
        assert(prctl(PR_SET_PDEATHSIG,SIGKILL)==0);if(getppid()!=parent_pid)_exit(3);
        close(ready[0]);close(gate[1]);Store writer(c);assert(writer.enabled());
        assert(writer.enqueue_file("interrupted",{9},MiB,[&](const std::string& target,std::string&)->bool {
            int fd=::open(target.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600),error=0;assert(fd>=0);
            assert(write_bytes(fd,MiB,error) && fsync(fd)==0);close(fd);
            // Exact barrier: complete durable payload exists, callback has NOT returned,
            // so the real Store cannot publish its catalogue entry yet.
            assert(::write(ready[1],"R",1)==1);char release;assert(::read(gate[0],&release,1)==1);return true;
        }));writer.flush();_exit(2); // Parent must kill us while blocked at the barrier.
    }
    close(ready[1]);close(gate[0]);pollfd event{ready[0],POLLIN,0};assert(poll(&event,1,5000)==1);
    char signal;assert(read(ready[0],&signal,1)==1 && signal=='R');
    assert(files_with(c.directory,".part")==1 && files_with(c.directory,".idx")==1);
    assert(kill(child,SIGKILL)==0);int status=0;assert(waitpid(child,&status,0)==child && WIFSIGNALED(status) && WTERMSIG(status)==SIGKILL);
    close(ready[0]);close(gate[1]);
    {
        Store recovered(c);assert(recovered.enabled());assert(!recovered.find("interrupted",{9}));assert(recovered.find("baseline",{8}));
        assert(recovered.enqueue("interrupted",{9},std::vector<uint8_t>(4096,9)));recovered.flush();
        assert(recovered.stats().writes==1 && recovered.stats().failures==0);
        assert(recovered.find("interrupted",{9}));assert(files_with(c.directory,".part")==0);
        auto budget=recovered.reserve(c.queue_bytes);assert(budget);
    }
    {Store fresh(c);assert(fresh.find("baseline",{8}) && fresh.find("interrupted",{9}));}
    std::cout<<"{\"status\":\"REAL_ENOSPC_AND_KILLED_WRITER_RECOVERY_PASS\",\"errno\":"<<observed_errno
             <<",\"writer_killed_before_catalogue\":true,\"fresh_lookup\":true,\"root\":\""<<run.string()<<"\"}\n";
}
