#include "prefill/short/policy.hpp"
#include <stdexcept>
#include <iostream>
using namespace strata::m31::shortprefill;
static void check(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
int main() {
    check(select(1)==Path::decode,"one token decode changed");
    for(int rows=2;rows<=31;++rows)check(select(rows)==Path::cpu,"CPU short chunk boundary");
    for(int rows=32;rows<=1024;++rows)check(select(rows)==Path::hybrid,"hybrid boundary");
    for(int rows : {1025,1536,2048,4096,16384,32768})check(select(rows)==Path::batch,"long chunk path changed");
    std::cout << "PASS fixed short-chunk routing boundaries\n";
}
