// CPU-only full GGUF geometry/tensor-map admission. No weights uploaded/hashed.
#include "ds41_geometry.hpp"
#include "strata/kernels/native_expert_registry.hpp"
#include <set>
#include <tuple>
#include <iostream>
using namespace strata;
using namespace strata::v41;
int main(int argc,char **argv) {
    if(argc!=2){std::cerr<<"usage: check_ds41_gguf_headers first-shard.gguf\n";return 2;}
    try {
        auto model=GgufModel::open(argv[1]);Ds41Geometry geometry;
        auto error=ds41_read_geometry(model,geometry);
        if(!error.empty()){std::cerr<<"geometry: "<<error<<'\n';return 1;}
        auto layers=ds41_file_layers(model,geometry);
        error=ds41_check_tensors(model,geometry,layers);
        if(!error.empty()){std::cerr<<"tensors: "<<error<<'\n';return 1;}
        std::set<std::tuple<int,int,int>> classes;
        int64_t checked=0;
        for(auto layer:layers) {
            const auto & ex=geometry.experts.at((size_t)layer);
            for(size_t expert=0;expert<ex.gu_type.size();++expert) {
                const int gate=ex.gu_type[expert], down=ex.d_type[expert];
                if(!strata::kernels::registry::supported_triple(gate,gate,down,geometry.n_embd,geometry.n_ff)) {
                    std::cerr<<"kernel admission: physical layer "<<layer<<" expert "<<expert
                             <<" types "<<gate<<'/'<<gate<<'/'<<down<<" dimensions "<<geometry.n_embd<<'x'<<geometry.n_ff<<'\n';
                    return 1;
                }
                classes.emplace(gate,gate,down);++checked;
            }
        }
        std::cout<<"GPU-registry CPU admission PASS experts="<<checked<<" classes="<<classes.size()<<'\n';
        for(const auto & types:classes)std::cout<<"class "<<std::get<0>(types)<<'/'<<std::get<1>(types)<<'/'<<std::get<2>(types)<<'\n';
        std::cout<<"PASS shards="<<model.size()<<" layers="<<layers.size()<<" context="<<geometry.ctx_len<<" vocab="<<geometry.vocab<<" PLE="<<ds41_is_ple_gguf(model)<<'\n';
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
