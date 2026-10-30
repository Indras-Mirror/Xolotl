// Opt-in native continuous batching. Each worker owns the existing request,
// cache, grammar and RNG state; a coordinator combines their main-model rows
// layer by layer. MTP proposals and verification remain enabled per owner.
#ifndef __HIP_PLATFORM_AMD__
class Q36BatchExecutor {
    struct Call {
        Engine* owner;
        int rows, heads;
        int64_t pos;
        bool verify;
        unsigned long long epoch;
        const float* const* images;
        std::promise<bool> completion;
        bool completed = false;
    };
    Engine& scratch;
    int device;
    int slots;
    std::function<size_t()> active_owners;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::shared_ptr<Call>> pending;
    bool stopping = false;
    std::thread thread;
    bool trace = std::getenv("STRATA_Q36_BATCH_TRACE") != nullptr;
    int wait_us = [] { const char* value=std::getenv("STRATA_Q36_BATCH_WAIT_US"); return value?std::max(0,std::min(5000,std::atoi(value))):150; }();
    void loop() {
        CK(cudaSetDevice(device));
        for (;;) {
            std::vector<std::shared_ptr<Call>> calls;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || !pending.empty(); });
                if (pending.empty() && stopping) return;
                // A short bounded rendezvous collects independently ready MTP
                // verify windows without waiting for an idle request slot.
                const size_t desired=std::min<size_t>((size_t)slots,std::max<size_t>(1,active_owners?active_owners():(size_t)slots));
                const auto deadline=Clock::now()+std::chrono::microseconds(wait_us);
                changed.wait_until(lock,deadline,[&] { return stopping || pending.size()>=desired; });
                int rows = 0;
                while (!pending.empty() && rows + pending.front()->rows <= scratch.tmax) {
                    auto call = pending.front(); pending.pop_front();
                    rows += call->rows; calls.push_back(std::move(call));
                }
            }
            std::vector<Engine::BatchGroup> groups;
            std::vector<const float*> images;
            int first = 0;
            for (auto& call : calls) {
                CK(cudaStreamSynchronize(call->owner->st));
                groups.push_back({call->owner, first, call->rows, call->pos, call->verify, call->heads});
                for (int row = 0; row < call->rows; ++row)
                    images.push_back(call->images ? call->images[row] : nullptr);
                first += call->rows;
            }
            bool ok = scratch.batch_forward(groups, images.data(), [&](Engine* canceled) {
                for (auto& call : calls) if (call->owner == canceled && !call->completed) {
                    call->completed = true; call->completion.set_value(false);
                }
            });
            CK(cudaStreamSynchronize(scratch.st));
            if (trace) {
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
                std::string slots, rows, epochs, canceled;
                bool verify = true;
                for (const auto& call : calls) {
                    if (!slots.empty()) { slots += ','; rows += ','; epochs += ','; canceled += ','; }
                    slots += std::to_string(call->owner->batch_slot);
                    rows += std::to_string(call->rows);
                    epochs += std::to_string(call->epoch);
                    canceled += call->completed ? "true" : "false";
                    verify = verify && call->verify;
                }
                std::fprintf(stderr, "Q36_BATCH_EVENT {\"event\":\"forward\",\"monotonic_ns\":%lld,\"slots\":[%s],\"rows\":[%s],\"request_epochs\":[%s],\"canceled\":[%s],\"verify\":%s,\"mtp_d\":%d}\n",
                             (long long) ns, slots.c_str(), rows.c_str(), epochs.c_str(), canceled.c_str(), verify ? "true" : "false", scratch.mtp_d);
            }
            for (auto& call : calls) if (!call->completed) call->completion.set_value(ok);
        }
    }
public:
    Q36BatchExecutor(Engine& engine, int device_, int slots_, std::function<size_t()> active):
        scratch(engine), device(device_), slots(slots_), active_owners(std::move(active)), thread([this] { loop(); }) {}
    ~Q36BatchExecutor() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        changed.notify_one(); thread.join();
    }
    bool submit(Engine& owner, int rows, int64_t pos, int heads, bool verify, const float* const* images) {
        auto call = std::make_shared<Call>();
        call->owner = &owner; call->rows = rows; call->pos = pos;
        call->heads = heads; call->verify = verify; call->epoch = owner.batch_epoch; call->images = images;
        auto result = call->completion.get_future();
        { std::lock_guard<std::mutex> lock(mutex); pending.push_back(call); }
        changed.notify_one();
        return result.get();
    }
};

int serve_parallel(Engine& scratch, ImageCache& images, const std::string& model,
                   int count, int snap_every, int prefill_chunk, size_t park_bytes, int snapshots,
                   strata::q36::kv::Budget& budget, strata::q36::kv::Driver& driver, bool warm) {
    int device = 0; CK(cudaGetDevice(&device));
    if (scratch.m->hybrid) { std::fprintf(stderr, "strata-q36: --batch needs resident immutable experts\n"); return 2; }
    auto host_hub=std::make_shared<strata::q36::cache::Hub>((size_t)std::max(0,snapshots));
    host_hub->track_page_versions=scratch.resident_prefix_reuse;
    struct Worker {
        std::unique_ptr<Engine> engine;
        std::mutex mutex;
        std::condition_variable changed;
        std::deque<std::string> input;
        bool eof = false, ready = false, busy = false, gpu_active = false, batch = false, admitted = false, control_locked = false;
        int slot = -1;
        long long produced = 0, prompt = 0, reused = 0, configured_max_new = 0;
        double pp_ms = 0;
        std::string deferred_done;
        Q36Channel channel;
        std::thread thread;
    };
    std::vector<std::unique_ptr<Worker>> workers;
    std::mutex output_mutex, admission_mutex, control_mutex;
    std::atomic<int> control_slot{0};
    std::condition_variable admission_changed;
    bool ending = false;
    auto output = [&](const std::string& line) {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::fwrite(line.data(), 1, line.size(), stdout); std::fflush(stdout);
    };
    Q36BatchExecutor executor(scratch, device, count, [&] {
        size_t active=0;
        for(auto& worker:workers) { std::lock_guard<std::mutex> lock(worker->mutex); if(worker->gpu_active) ++active; }
        return active;
    });
    for (int slot = 0; slot < count; ++slot) {
        auto owner = std::make_unique<Worker>();
        owner->slot = slot;
        owner->engine = std::make_unique<Engine>();
        Engine& engine = *owner->engine;
        engine.kv_budget = &budget; engine.kv_driver = &driver;
        engine.use_graph = false; engine.dynamic_attention = true;
        engine.resident_prefix_reuse=scratch.resident_prefix_reuse;
        engine.resident_prefix_verify=scratch.resident_prefix_verify;
        engine.shared_prefill=scratch.shared_prefill;
        engine.cache_hub=host_hub;
        engine.init(*scratch.m, scratch.ctx, prefill_chunk);
        engine.alloc_prefill_rollback(); engine.reset();
        engine.mtp_d = scratch.mtp_d; engine.max_snaps = std::max(0, snapshots);
        engine.batch_slot = slot;
        if (warm) engine.warmup();
        CK(cudaStreamSynchronize(engine.st));
        if (!engine.kv_arena->trim(0)) return 1;
        Worker* worker = owner.get();
        engine.batch_dispatch = [&, worker](int rows, int64_t pos, int heads, bool verify, const float* const* pic) {
            return executor.submit(*worker->engine, rows, pos, heads, verify, pic);
        };
        engine.batch_begin = [&, worker] {
            control_mutex.lock(); worker->control_locked = true;
            control_slot.store(worker->slot);
            ++worker->engine->batch_epoch;
        };
        engine.batch_parsed = [&,worker](long long max_new,size_t prompt_tokens) {
            worker->configured_max_new=max_new;
            if(std::getenv("STRATA_Q36_BATCH_TRACE")) {
                auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
                std::fprintf(stderr,"Q36_BATCH_EVENT {\"event\":\"begin\",\"monotonic_ns\":%lld,\"slot_id\":%d,\"request_epoch\":%llu,\"max_new\":%lld,\"prompt_tokens\":%zu,\"mtp_d\":%d}\n",(long long)ns,worker->slot,worker->engine->batch_epoch,max_new,prompt_tokens,worker->engine->mtp_d);
            }
        };
        engine.batch_admit = [&, worker](size_t end) {
            std::unique_lock<std::mutex> lock(admission_mutex);
            // One request retains its configured 256K limit. Simultaneous
            // requests share this physical budget and wait rather than truncate.
            if (worker->engine->kv_arena->bytes_for(end) > budget.cap()) return false;
            for(auto& candidate:workers) {
                std::lock_guard<std::mutex> state_lock(candidate->mutex);
                if(candidate.get()!=worker && !candidate->gpu_active)
                    if(!candidate->engine->resize_attention(1)) return false;
            }
            if(!worker->engine->resize_attention(end)) return false;
            auto commit_admission=[&] {
                bool ok=worker->engine->ensure_kv(end);
                if(ok) { std::lock_guard<std::mutex> state_lock(worker->mutex); worker->gpu_active=true; }
                return ok;
            };
            for (;;) {
                const size_t target=worker->engine->kv_arena->bytes_for(end);
                const size_t held=worker->engine->kv_arena->committed_bytes();
                const size_t delta=target>held?target-held:0;
                // Enough accounting space plus a driver allocation failure is
                // a hard refusal. Waiting cannot cure a failed CUDA mapping.
                if (delta <= budget.cap()-budget.used())
                    return commit_admission();
                for (auto& candidate : workers) {
                    std::lock_guard<std::mutex> state_lock(candidate->mutex);
                    if (candidate.get() != worker && candidate->gpu_active) continue;
                    CK(cudaStreamSynchronize(candidate->engine->st));
                    if (!candidate->engine->kv_arena->trim(0)) std::abort();
                }
                const size_t reclaimed_held=worker->engine->kv_arena->committed_bytes();
                const size_t needed=target>reclaimed_held?target-reclaimed_held:0;
                if (needed <= budget.cap()-budget.used())
                    return commit_admission();
                if (ending || (worker->engine->cancel && worker->engine->cancel->load())) return false;
                admission_changed.wait_for(lock, std::chrono::milliseconds(2));
            }
            return true;
        };
        auto finish = [&, worker]() {
            CK(cudaStreamSynchronize(worker->engine->st));
            // Keep idle mappings charged to the common budget. A later
            // admission may reclaim them after this owner is visibly idle.
            admission_changed.notify_all();
            std::string done;
            unsigned long long released_epoch=0;
            long long released_ns=0, released_generated=0;
            {
                std::lock_guard<std::mutex> lock(worker->mutex);
                done.swap(worker->deferred_done);
                released_epoch=worker->engine->batch_epoch; released_generated=worker->produced;
                worker->gpu_active=false; worker->busy=false;
                released_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
            }
            if(std::getenv("STRATA_Q36_BATCH_TRACE"))
                std::fprintf(stderr,"Q36_BATCH_EVENT {\"event\":\"slot_released\",\"monotonic_ns\":%lld,\"slot_id\":%d,\"request_epoch\":%llu,\"generated\":%lld}\n",released_ns,worker->slot,released_epoch,released_generated);
            if (!done.empty()) output(done);
            if (worker->control_locked) { worker->control_locked = false; control_mutex.unlock(); }
        };
        engine.batch_idle = finish;
        worker->channel.read = [worker](std::string& line) {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->changed.wait(lock, [&] { return worker->eof || !worker->input.empty(); });
            if (worker->input.empty()) return false;
            line = std::move(worker->input.front()); worker->input.pop_front(); return true;
        };
        worker->channel.write = [&, worker](const std::string& line) {
            if (line.rfind("READY", 0) == 0) {
                { std::lock_guard<std::mutex> lock(worker->mutex); worker->ready=true; }
                worker->changed.notify_all(); return;
            }
            if (line.rfind("INFO", 0) == 0) return;
            if (line.rfind("RESUME ", 0) == 0) worker->reused = std::strtoll(line.c_str()+7, nullptr, 10);
            if (line.rfind("PP ", 0) == 0) {
                long long read = 0, total = 0; double ms = 0;
                std::sscanf(line.c_str(), "PP %lld %lld %lf", &read, &total, &ms);
                worker->prompt = total; worker->pp_ms = ms;
            }
            if (line.rfind("T ", 0) == 0) {
                ++worker->produced;
                if (!worker->batch) { output(line); return; }
                const int token = std::atoi(line.c_str()+2);
                if (std::getenv("STRATA_Q36_BATCH_TRACE")) {
                    auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
                    std::fprintf(stderr,"Q36_BATCH_EVENT {\"event\":\"progress\",\"monotonic_ns\":%lld,\"slot_id\":%d,\"request_epoch\":%llu,\"generated\":%lld,\"mtp_enabled\":%s}\n",(long long)ns,worker->slot,worker->engine->batch_epoch,worker->produced,worker->engine->mtp_d>0?"true":"false");
                }
                if (!worker->admitted) {
                    worker->admitted = true;
                    output(line + "DONE 1 " + std::to_string(worker->prompt) + " " + std::to_string(worker->pp_ms) + " 0 length 0 0 " + std::to_string(worker->reused) + "\nBADM " + std::to_string(worker->slot) + " 1\n");
                    if (worker->control_locked) { worker->control_locked = false; control_mutex.unlock(); }
                } else output("BT " + std::to_string(worker->slot) + " " + std::to_string(token) + "\n");
                return;
            }
            if (line.rfind("DONE ", 0) == 0) {
                long long produced = 0, prompt = 0; double pp = 0, decode = 0; char reason[32] = {};
                long long accepted=0,offered=0,cached=0;
                std::sscanf(line.c_str(), "DONE %lld %lld %lf %lf %31s %lld %lld %lld", &produced, &prompt, &pp, &decode, reason,&accepted,&offered,&cached);
                if(std::getenv("STRATA_Q36_BATCH_TRACE")) {
                    auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
                    std::fprintf(stderr,"Q36_BATCH_EVENT {\"event\":\"finish\",\"monotonic_ns\":%lld,\"slot_id\":%d,\"request_epoch\":%llu,\"generated\":%lld,\"mtp_d\":%d,\"accepted\":%lld,\"offered\":%lld,\"cached\":%lld,\"finish\":\"%s\"}\n",(long long)ns,worker->slot,worker->engine->batch_epoch,produced,worker->engine->mtp_d,accepted,offered,cached,reason);
                }
                std::lock_guard<std::mutex> lock(worker->mutex);
                if (worker->batch && worker->admitted)
                    worker->deferred_done = "BDONE " + std::to_string(worker->slot) + " " + std::to_string(produced) + " " + reason + " " + std::to_string(decode) + " accepted=" + std::to_string(accepted) + " offered=" + std::to_string(offered) + "\n";
                else worker->deferred_done = line + (worker->batch ? "BADM " + std::to_string(worker->slot) + " 0\n" : "");
                return;
            }
            if (line.rfind("ERR", 0) == 0) {
                std::lock_guard<std::mutex> lock(worker->mutex);
                // A foreground refusal belongs to its held control admission.
                // Once BADM was sent, errors must be slot-scoped. Publish only
                // after batch_idle has fenced/parked this owner; never make it
                // reclaimable from this output callback.
                worker->deferred_done = worker->batch && worker->admitted
                    ? "BDONE " + std::to_string(worker->slot) + " " + std::to_string(worker->produced) + " error 0\n"
                    : line;
                return;
            }
            output(line);
        };
        workers.push_back(std::move(owner));
    }
    if(!workers.empty()) {
        workers.front()->engine->alloc_snaps();
        workers.front()->engine->alloc_pages(park_bytes);
    }
    size_t physical_free=0, physical_total=0;
    CK(cudaMemGetInfo(&physical_free,&physical_total));
    if (physical_free < budget.cap() + (size_t(256)<<20)) {
        std::fprintf(stderr,"strata-q36: batch weights/scratch leave %.0f MiB free, physical KV cap %zu MiB plus256MiB reserve cannot fit; refusing READY\n", physical_free/1048576.0,budget.cap()>>20);
        return 2;
    }
    std::fprintf(stderr,"strata-q36: batch physical preflight free=%.0fMiB cap=%zuMiB reserve=256MiB max_request=%d slots=%d MTP=%d\n",physical_free/1048576.0,budget.cap()>>20,scratch.ctx,count,scratch.mtp_d);
    for (auto& owner : workers) {
        Worker* worker = owner.get();
        worker->thread = std::thread([&, worker] {
            CK(cudaSetDevice(device)); q36_channel = &worker->channel;
            serve(*worker->engine, images, snap_every, park_bytes, model);
            q36_channel = nullptr;
        });
    }
    for(auto& worker:workers) {
        std::unique_lock<std::mutex> lock(worker->mutex);
        worker->changed.wait(lock,[&] { return worker->ready; });
    }
#ifdef STRATA_Q36_GRAMMAR
    output("INFO grammar=1\n");
#endif
    output("INFO engine=q36 img_refs=1 n_embd=" + std::to_string(scratch.m->g.n_embd) +
           " slot_cache=" + std::to_string(park_bytes>0 && snapshots>0 ? 1 : 0) +
           " host_cache_shared=1 snapshots=" + std::to_string(host_hub->snaps.size()) +
           " host_park_mib=" + std::to_string((host_hub->n_pages*host_hub->page_bytes)>>20) +
           (scratch.resident_prefix_reuse ? " resident_prefix_reuse=1 checkpoint_clock=1" : "") +
           " batch_prefill_chunk=" + std::to_string(prefill_chunk) +
           " batch_slots=" + std::to_string(count) + " batch_mtp=1 batch_pinned=1 max_request_context=" + std::to_string(scratch.ctx) + " aggregate_kv_mib=" + std::to_string(budget.cap() >> 20) + "\nREADY " + std::to_string(scratch.ctx) + " stop\n");
    int foreground = 0;
    bool checkpoint_paused=false;
    Clock::time_point checkpoint_begin;
    auto enqueue = [&](int slot, std::string line) {
        Worker& worker = *workers[(size_t) slot];
        { std::lock_guard<std::mutex> lock(worker.mutex); worker.input.push_back(std::move(line)); }
        worker.changed.notify_one();
    };
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "QUIT") break;
        if(scratch.resident_prefix_reuse && (line=="CHECKPOINT_PAUSE" || line=="CHECKPOINT_RESUME")) {
            // Same idle protocol used by the optional checkpoint frontend. A
            // pause invalidates only the CPU shortcut, not the parked cache.
            std::unique_lock<std::mutex> admission_lock(admission_mutex,std::try_to_lock);
            if(!admission_lock.owns_lock()){output("ERR checkpoint: admission busy\n");continue;}
            std::vector<std::unique_lock<std::mutex>> owner_locks;
            for(auto& owner:workers)owner_locks.emplace_back(owner->mutex);
            bool busy=false;
            for(auto& owner:workers)busy=busy || owner->busy || owner->gpu_active;
            if(busy){output("ERR checkpoint: active request\n");continue;}
            if(line=="CHECKPOINT_PAUSE") {
                for(auto& owner:workers){owner->engine->resident_prefix.invalidate();owner->engine->resident_ram_equal_end=0;}
                if(!checkpoint_paused)checkpoint_begin=Clock::now();
                checkpoint_paused=true;
                output("CHECKPOINT_PAUSED graph_execs=0\n");
            } else {
                if(!checkpoint_paused){output("ERR checkpoint: not paused\n");continue;}
                const auto paused_us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-checkpoint_begin).count();
                checkpoint_paused=false;
                output("CHECKPOINT_RESUMED paused_us="+std::to_string(paused_us)+" graph_execs=0\n");
            }
            continue;
        }
        if (line == "STOP") { enqueue(control_slot.load(), "STOP"); continue; }
        if (line.rfind("BSTOP ",0)==0) {
            int slot = std::atoi(line.c_str()+6); if(slot>=0 && slot<count) enqueue(slot,"STOP"); continue;
        }
        if (line.rfind("BYIELD",0)==0) continue;
        if (line.rfind("HAVE",0)==0) {
            std::lock_guard<std::recursive_mutex> cache_lock(images.mutex);
            std::string response="HAVE"; bool first=true;
            std::string keys=line.size()>5?line.substr(5):""; size_t at=0;
            while(at<keys.size()) {
                size_t end=keys.find(',',at); std::string key=keys.substr(at,end==std::string::npos?end:end-at);
                char* tail=nullptr; uint64_t value=std::strtoull(key.c_str(),&tail,16);
                auto* image=!key.empty() && tail && !*tail?images.find(value):nullptr;
                response+=first?" ":","; first=false;
                response+=image?std::to_string(image->n)+":"+std::to_string(image->nx)+":"+std::to_string(image->ny):"0";
                if(end==std::string::npos)break; at=end+1;
            }
            output(response+"\n"); continue;
        }
        if(checkpoint_paused){output("ERR checkpoint: paused\n");continue;}
        const bool bimg = line.rfind("BGENI ",0)==0, bgen = line.rfind("BGEN ",0)==0;
        const bool img = line.rfind("GENI ",0)==0, gen = line.rfind("GEN ",0)==0;
        int slot = 0;
        if (bimg || bgen) {
            char* end = nullptr; long value = std::strtol(line.c_str()+(bimg?6:5), &end,10);
            if (value<0 || value>=count || !end || *end!=' ') { output("ERR BGEN: invalid slot\n"); continue; }
            slot=(int)value; line=std::string(bimg?"GENI":"GEN")+end;
        } else if (!(img || gen)) { output("ERR expected GEN/BGEN/HAVE/BSTOP\n"); continue; }
        Worker& worker=*workers[(size_t)slot];
        {
            std::lock_guard<std::mutex> lock(worker.mutex);
            if(worker.busy) { output("ERR BGEN: no such free slot\n"); continue; }
            worker.busy=true; worker.batch=bimg||bgen; worker.admitted=false;
            worker.produced=0; worker.prompt=0; worker.reused=0; worker.pp_ms=0;
        }
        worker.prompt = (long long) parse_ids(line.substr(line.find_last_of(' ')+1).c_str()).size();
        foreground=slot; enqueue(slot,std::move(line));
    }
    { std::lock_guard<std::mutex> lock(admission_mutex); ending=true; }
    admission_changed.notify_all();
    for(auto& worker:workers) { enqueue(worker->slot,"STOP"); {std::lock_guard<std::mutex> lock(worker->mutex);worker->eof=true;} worker->changed.notify_all(); }
    for(auto& worker:workers) worker->thread.join();
    return 0;
}
#endif
