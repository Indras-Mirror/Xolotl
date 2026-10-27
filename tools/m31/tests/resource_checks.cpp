#include "../m31_config.hpp"
#include "../prefill/chunk_limit.hpp"
#include "../prefill/plan.hpp"
#include "../prefill/slot_leases.hpp"
#include "../prefill/expert_budget.hpp"
#include <climits>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <thread>
#include <future>

using namespace strata::m31;
static void expect(bool value,const char * why) { if(!value)throw std::runtime_error(why); }
template<class F> static void rejects(F f,const char * why) { bool failed=false;try{f();}catch(const std::exception&){failed=true;}expect(failed,why); }
int main(int argc,char **argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--environment") {
            prepare_engine_environment();validate_config(Config{});
            std::printf("fusion=%s\n",std::getenv("GGML_CUDA_DISABLE_FUSION"));
            std::puts(effective_config(Config{}).c_str());
            return 0;
        }

        // Mandatory independent of the optional VRAM planner or backend choice.
        for(int64_t position : {0LL,255LL,256LL,32768LL,1048576LL}) {
            const int n=msa_chunk_limit(position,32768,16,128);
            expect(n>=0 && n<=32768,"chunk cap outside request");
            if(n)expect(uint64_t((position+n+255)/256*256)*16*uint64_t(n)<INT32_MAX,"accepted chunk overflows int32");
            if(n<32768)expect(uint64_t((position+n+1+255)/256*256)*16*uint64_t(n+1)>=INT32_MAX,"chunk cap is not maximal");
        }
        expect(msa_chunk_limit(INT32_MAX,1,INT32_MAX,128)==0,"impossible one-row request accepted");
        rejects([]{msa_chunk_limit(-1,1,16,128);},"negative position accepted");
        rejects([]{msa_chunk_limit(0,1,16,129);},"invalid MSA block accepted");
        rejects([]{prefill_product(UINT64_MAX,2);},"multiplication overflow accepted");
        rejects([]{prefill_sum(UINT64_MAX,1);},"addition overflow accepted");
        Geometry g;
        g.embd=6144;g.expert_ff=1536;g.dense_ff=16384;g.heads=48;g.head_dim=128;
        g.index_heads=16;g.index_dim=128;g.used=4;g.experts=128;g.block=128;g.shared=1;
        PrefillBudget b;
        b.free_after_resident=uint64_t(32)<<30;b.max_chunk=128;b.context=4096;b.remaining=128;
        b.streaming_slots=256;b.native_gpu_ffn=true;b.gpu_prefill_chain=true;b.top_k_mask=true;b.inplace_score=true;
        b.retained_dense_bytes=uint64_t(2)<<30;b.retained_routed_bytes=uint64_t(3)<<30;
        auto plan=plan_prefill(g,b);
        expect(plan.chunk==128,"reasonable native plan rejected");
        expect(plan.scratch_estimate==plan.persistent_scratch_estimate+plan.dense_scratch_estimate+plan.routed_scratch_estimate,
            "coexisting retained owners must be summed");
        b.free_after_resident=(uint64_t(4)<<30)+b.reserve_bytes+b.extra_scratch_bytes;
        expect(plan_prefill(g,b).chunk==0,"planner budget mistakenly uses max of retained owners");
        b.free_after_resident=uint64_t(32)<<30;g.block=129;
        rejects([&]{plan_prefill(g,b);},"nondividing MSA block accepted");
        // Force an uploader to attempt reuse while a real asynchronous reader is
        // blocked. Owner state is serialized, matching GPU cache bookkeeping.
        SlotLeases slots;slots.reset(2);slots.reserve_write(0);
        rejects([&]{slots.begin_read(0);},"unpublished DMA admitted a reader");
        slots.publish_write(0);slots.begin_read(0);
        std::promise<void> finish;auto ready=finish.get_future();
        auto reader=std::async(std::launch::async,[&]{ready.wait();});
        rejects([&]{slots.reserve_write(0);},"active reader overwritten");
        slots.reserve_write(1);rejects([&]{slots.reserve_write(1);},"concurrent writers accepted");
        finish.set_value();reader.get();slots.finish_reads();slots.reserve_write(0);slots.publish_write(0);
        slots.reset(2);slots.reserve_write(1);slots.publish_write(1);slots.begin_read(1);slots.finish_reads();
        expect(slots.writable(1),"recovery failed to release leases");
        expect(choose_expert_budget(24000,10,128)==2560,"automatic budget exceeds two layers");
        expect(choose_expert_budget(4500,10,128)==1500,"automatic budget must leave two thirds free");
        rejects([]{choose_expert_budget(3000,10,128);},"automatic budget admitted a partial layer");
        rejects([]{choose_expert_budget(UINT64_MAX,UINT64_MAX,128);},"automatic budget overflow accepted");
        Config c;validate_config(c);
        c.persistent_cpu_pool=false;rejects([&]{validate_config(c);},"unsafe fused pool combination accepted");
        c=Config{};c.register_arena=true;rejects([&]{validate_config(c);},"unbudgeted pinning accepted");
        c=Config{};c.upload_streams=5;rejects([&]{validate_config(c);},"invalid copy-stream count accepted");
        std::puts("M31 CPU resource checks passed");
        return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
