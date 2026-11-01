#include "strata/program/batch_cancel.hpp"
#include <cassert>
#include <thread>
using namespace strata::program;
int main(){
 BatchStop s;assert(parse_batch_stop("BSTOP 3 42",s)&&s.slot==3&&s.epoch==42);
 for(auto x:{"BSTOP 3","BSTOP -1 1","BSTOP 0 0","BSTOP 0 1x","BSTOP 0 1 ","BSTOP 0 18446744073709551616"})assert(!parse_batch_stop(x,s));
 assert(batch_cancel_epoch("BGEN 0 42 cancel_epoch=300 1,2")==300);
 BatchCancelMailbox m(4);m.request(1,100);m.begin(1,100);assert(m.cancelled(1,100));
 m.retire(1,100);assert(!m.cancelled(1,100));m.begin(1,101);m.request(1,100);assert(!m.cancelled(1,101));
 m.request(1,101);m.request(1,100);assert(m.cancelled(1,101));m.retire(1,100);assert(m.cancelled(1,101));
 m.begin(2,200);m.request(1,200);assert(!m.cancelled(2,200));
 m.request(2,201);m.request(2,200);assert(m.cancelled(2,200));m.retire(2,200);m.begin(2,201);assert(m.cancelled(2,201));
 std::thread a([&]{for(uint64_t i=1;i<=100000;i++)m.request(3,i);});
 std::thread b([&]{for(uint64_t i=100000;i>0;i--)m.request(3,i);});a.join();b.join();
 m.begin(3,100000);assert(m.cancelled(3,100000));m.begin(3,100001);assert(!m.cancelled(3,100001));
 m.request(-1,1);m.request(9,1);assert(!m.cancelled(9,1));
}
