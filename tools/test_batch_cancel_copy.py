from pathlib import Path
import subprocess, tempfile, os
temporary=tempfile.TemporaryDirectory(prefix='strata-copy-cancel-');root=Path(temporary.name)
s=(Path(__file__).resolve().parents[1]/'src/program/generate.cpp').read_text(); start=s.index('        auto copy_to_slot ='); end=s.index('        // slot b\'s sessions',start); code=s[start:end]
pre=r'''
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <future>
#include <chrono>
#include <thread>
int ops=0,fail_at=-1,joins=0; std::future<void> pending;
bool operation(std::string& e){++ops;if(ops==fail_at){e="GPU ERROR";return false;}return true;}
int cudaSuccess=0; int cudaDeviceSynchronize(){std::string e;return operation(e)?0:1;}
namespace strata {namespace core {
struct SessionState{int qsa_alloc=3,qsa_ord0=0;std::vector<int> qsa_states{0,1,2};};
struct OnDevice{OnDevice(int){}};
struct ConversationCheckpoint{std::vector<int32_t> ids;std::vector<char> gdn;};
struct ConversationKv{};
void conversation_checkpoint_join(void* p){assert(p);++joins;if(pending.valid())pending.get();}
bool conversation_checkpoint_save(ConversationCheckpoint& c,SessionState&,int,std::string& e){if(!operation(e))return false;c.gdn.resize(8);char* p=c.gdn.data();pending=std::async(std::launch::async,[p]{std::this_thread::sleep_for(std::chrono::milliseconds(2));p[0]=9;});return true;}
bool conversation_checkpoint_restore(ConversationCheckpoint&,SessionState&,int,std::string& e){return operation(e);}
bool conversation_kv_save(ConversationKv&,int&,int,int64_t,bool,std::string& e){return operation(e);}
bool conversation_kv_restore(ConversationKv&,int&,int,int64_t,bool,std::string& e){return operation(e);}
}}
struct Draft {int kv=0;int device(){return 0;}bool idle(std::string&e){return operation(e);}int& kv_state(){return kv;}void set_prompt_len(int64_t){}};
int main(){
 strata::core::SessionState ss,target; struct Stage{strata::core::SessionState ss;int dev=0;}; std::vector<std::unique_ptr<Stage>> stages;
 std::vector<std::vector<strata::core::SessionState*>> bslot_ss{{&target}};
 bool slot_copy_direct=false,batch_mtp=true;int g=0,draft_geometry=0; Draft mtp,draft;std::vector<Draft*>slot_mtp{&draft};
 auto direct_slot_copy=[&](bool,int,const std::vector<int32_t>&,std::string&e){return operation(e);};
'''
post=r'''
 std::vector<int32_t> ids{1,2,3};
 for(int limit=0;limit<22;++limit){ops=0;fail_at=-1;joins=0;std::string e;bool cancelled=false;bool ok=copy_to_slot(0,ids,e,[&]{return ops>=limit;},&cancelled); bool cached=true,tainted=false,cont=true;
 if(!ok){cached=false;if(cancelled){tainted=true;e.clear();}cont=false;}
 if(cancelled){assert(!cont&&!cached&&tainted&&e.empty());assert(ops<=limit+5);}else assert(ok);
 if(pending.valid())pending.get();
 }
 for(int f=1;f<=8;++f){ops=0;fail_at=f;std::string e;bool c=true;bool ok=copy_to_slot(0,ids,e,[]{return false;},&c);assert(!ok&&!c);if(pending.valid())pending.get();}
 ops=0;fail_at=-1;std::string e;bool c=true;assert(copy_to_slot(0,ids,e,[]{return false;},&c)&&!c);int normal=ops;if(pending.valid())pending.get();
 ops=0;batch_mtp=true;assert(copy_to_slot(0,ids,e,[]{return false;},&c));assert(ops==normal);if(pending.valid())pending.get();

 { struct Slot{bool cached=true,tainted=false;};std::vector<Slot> bs(1);int admit_slot=0;bool cont=true;bool batch_cancel_enabled=true;std::string err;std::vector<int32_t>live{1};auto request_cancelled=[]{return false;};auto copy_to_slot=[](int,const std::vector<int32_t>&,std::string&,const std::function<bool()>&,bool*c){*c=true;return false;};
__ACTUAL_ADMISSION_TRANSITION__

assert(!cont&&!bs[0].cached&&bs[0].tainted&&err.empty());}
 puts("PASS extracted copy: before/partial/every boundary/normal/error/async scratch");
}
'''
a=s.index('                bool copy_cancelled = false;');b=s.index('                const auto tc1',a)
post=post.replace('__ACTUAL_ADMISSION_TRANSITION__',s[a:b])
p=root/'test_copy.cpp';p.write_text(pre+code+post)
subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-g','-O1','-fsanitize=address,undefined','-pthread',str(p),'-o',str(root/'test_copy')],check=True)
subprocess.run([str(root/'test_copy')],check=True)
