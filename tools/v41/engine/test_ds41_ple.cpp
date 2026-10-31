// Modern DeepSeek-V4.1 PLE metadata and combined Q5_K Engram reader, without CUDA/model downloads.
#include "ds41_engram.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
using namespace strata;
using namespace strata::v41;
namespace {
struct Bytes {
    std::vector<uint8_t> b;
    template<class T> void put(T v) { const auto * p = reinterpret_cast<const uint8_t *>(&v); b.insert(b.end(), p, p + sizeof v); }
    void str(const std::string & s) { put<uint64_t>(s.size()); b.insert(b.end(), s.begin(), s.end()); }
};
struct Key { std::string n; uint32_t type; Bytes value; };
struct Fixture {
    std::vector<Key> keys;
    std::vector<TensorInfo> ts;
    std::vector<std::vector<uint8_t>> payloads;
    void scalar(const std::string & k, int64_t v) { Bytes b; b.put(v); keys.push_back({k,11,std::move(b)}); }
    void real(const std::string & k, float v) { Bytes b; b.put(v); keys.push_back({k,6,std::move(b)}); }
    void string(const std::string & k, const std::string & v) { Bytes b; b.str(v); keys.push_back({k,8,std::move(b)}); }
    void array(const std::string & k, const std::vector<int64_t> & v) {
        Bytes b; b.put<uint32_t>(11); b.put<uint64_t>(v.size()); for (auto x:v) b.put(x); keys.push_back({k,9,std::move(b)});
    }
    void set_array(const std::string & k, const std::vector<int64_t> & v) {
        for (auto i=keys.begin();i!=keys.end();++i) { if(i->n==k) { keys.erase(i); break; } }
        array(k,v);
    }
    void tensor(const std::string & n, const std::vector<uint64_t> & sh, uint32_t type=0) {
        TensorInfo t; t.name=n; t.shape=sh; t.type=type; ts.push_back(t); payloads.emplace_back((size_t)tensor_payload_bytes(t),0);
    }
    void write(const std::string & path) {
        Bytes b; b.put<uint32_t>(0x46554747); b.put<uint32_t>(3); b.put<uint64_t>(ts.size()); b.put<uint64_t>(keys.size());
        for (const auto & k:keys) { b.str(k.n); b.put(k.type); b.b.insert(b.b.end(),k.value.b.begin(),k.value.b.end()); }
        uint64_t off=0;
        for (size_t i=0;i<ts.size();++i) { ts[i].offset=off; b.str(ts[i].name);b.put<uint32_t>(ts[i].shape.size());for(auto d:ts[i].shape)b.put(d);b.put(ts[i].type);b.put(off);off=(off+payloads[i].size()+31)&~31ull; }
        while(b.b.size()%32)b.b.push_back(0);
        for(size_t i=0;i<ts.size();++i){b.b.insert(b.b.end(),payloads[i].begin(),payloads[i].end());while(b.b.size()%32)b.b.push_back(0);}
        std::ofstream f(path,std::ios::binary);f.write((char*)b.b.data(),b.b.size());if(!f)throw std::runtime_error("fixture write");
    }
};
std::vector<uint8_t> q5row(int row) {
    std::vector<uint8_t> b(176); b[1]=0x38; b[3]=0x34; // d=.5, dmin=.25
    for(int g=0;g<8;++g){uint8_t sc=(uint8_t)(1+(row+g*7)%63),mn=(uint8_t)((row*3+g*5)%64);
        if(g<4){b[4+g]=sc;b[8+g]=mn;}else{b[8+g]=(sc&15)|((mn&15)<<4);b[g]|=(sc>>4)<<6;b[4+g]|=(mn>>4)<<6;}}
    for(int j=0;j<32;++j)b[16+j]=(uint8_t)(j*11+row*7);
    for(int j=0;j<128;++j)b[48+j]=(uint8_t)(j*13+row*17);
    return b;
}
float expected(int row,int j){int g=j/32,k=j%32;uint8_t ql=(uint8_t)((g/2*32+k)*13+row*17),qh=(uint8_t)(k*11+row*7);
    int q=((g&1)?ql>>4:ql&15)+((qh>>g)&1)*16;return .5f*(1+(row+g*7)%63)*q-.25f*((row*3+g*5)%64);}
Fixture fixture(){Fixture f;f.string("general.architecture","deepseek41");const std::string p="deepseek41.";
    for(auto kv:std::vector<std::pair<std::string,int64_t>>{{"embedding_length",256},{"block_count",2},{"attention.head_count",1},{"attention.key_length",4},{"rope.dimension_count",2},{"attention.q_lora_rank",4},{"attention.output_lora_rank",4},{"attention.output_group_count",1},{"attention.sliding_window",4},{"expert_count",1},{"expert_used_count",1},{"expert_feed_forward_length",256},{"expert_shared_count",1},{"hyper_connection.count",1},{"hyper_connection.sinkhorn_iterations",2},{"attention.indexer.head_count",1},{"attention.indexer.key_length",4},{"attention.indexer.top_k",2},{"attention.indexer.candidate_source_layer",-1},{"attention.indexer.candidate_top_k_blocks",0},{"attention.indexer.candidate_block_size",0},{"context_length",16},{"ple.heads_per_ngram",1},{"embedding_length_per_layer_input",256},{"ple.ngram_size",2},{"engram.pad_token_id",2},{"engram.compressed_vocab_size",4},{"expert_weights_norm",1},{"expert_gating_func",4}})f.scalar(p+kv.first,kv.second);
    for(auto kv:std::vector<std::pair<std::string,float>>{{"expert_weights_scale",1},{"hyper_connection.epsilon",1e-6f},{"attention.layer_norm_rms_epsilon",1e-5f},{"rope.freq_base",10000},{"attention.compress_rope_freq_base",160000},{"rope.scaling.factor",16},{"rope.scaling.yarn_beta_fast",32},{"rope.scaling.yarn_beta_slow",1},{"rope.scaling.original_context_length",16}})f.real(p+kv.first,kv.second);
    f.array(p+"swiglu_clamp_exp",{10,10});f.array(p+"swiglu_clamp_shexp",{10,10});f.array(p+"attention.compress_ratios",{0,0});f.array(p+"attention.kv_source_layers",{});f.array(p+"attention.indexer.source_layers",{});f.array(p+"ple.layers",{0,1});f.array(p+"ple.head_vocab_sizes",{5,7});f.array(p+"ple.head_offsets",{0,5});f.array(p+"ple.layer_multipliers",{3,5,7,11});f.array(p+"engram.token_map",{0,3,1,2});
    f.tensor("token_embd.weight",{256,4});f.tensor("output.weight",{256,4});f.tensor("output_norm.weight",{256});
    for(int l=0;l<2;++l){std::string b="blk."+std::to_string(l)+".";
        for(const char * site:{"attn","ffn"}){f.tensor(b+"hc_"+site+"_fn.weight",{256,3});f.tensor(b+"hc_"+site+"_base.weight",{3});f.tensor(b+"hc_"+site+"_scale.weight",{3});f.tensor(b+site+"_norm.weight",{256});}
        f.tensor(b+"attn_sinks.weight",{1});f.tensor(b+"attn_q_a.weight",{256,4});f.tensor(b+"attn_q_b.weight",{4,4});f.tensor(b+"attn_q_a_norm.weight",{4});f.tensor(b+"attn_kv.weight",{256,4});f.tensor(b+"attn_kv_a_norm.weight",{4});f.tensor(b+"attn_output_a.weight",{4,4,1});f.tensor(b+"attn_output_b.weight",{4,256});f.tensor(b+"ffn_gate_inp.weight",{256,1});f.tensor(b+"exp_probs_b.bias",{1});f.tensor(b+"exp_probs_b_vl.bias",{1});
        for(const char * role:{"gate","up","down"}) {f.tensor(b+"ffn_"+role+"_shexp.weight",{256,256});f.tensor(b+"ffn_"+role+"_exps.weight",{256,256,1},std::string(role)=="down"?(l?21:39):(l?18:16));}
        f.tensor(b+"engram_gate.weight",{256,1});f.tensor(b+"engram_kv.weight",{256,512});}
    f.tensor("per_layer_token_embd.weight",{256,12},13);for(int r=0;r<12;++r){auto b=q5row(r);std::copy(b.begin(),b.end(),f.payloads.back().begin()+r*176);}return f;}
void require(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
}
int main(){try{
    { MetaValue v; int64_t n = 0; v.type=MetaType::I64;v.u=(uint64_t)-1;require(detail::int64_value(v,n)&&n==-1,"signed metadata -1");
      v.u=INT64_MAX;require(detail::int64_value(v,n)&&n==INT64_MAX,"exact int64 maximum");
      v.type=MetaType::U64;v.u=UINT64_MAX;require(!detail::int64_value(v,n),"oversized uint64 accepted");
      v.type=MetaType::F64;v.f=0x1p63;require(!detail::int64_value(v,n),"float int64 overflow accepted");
      v.f=.5;require(!detail::int64_value(v,n),"fractional integer accepted"); }
    const auto temp = std::filesystem::temp_directory_path() /
        ("test-ds41-ple-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    require(std::filesystem::create_directory(temp), "temporary directory");
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(p, ec); } } cleanup{temp};
    const std::string path = (temp / "fixture.gguf").string();
    Fixture f=fixture();f.write(path);Ds41Geometry g;{auto m=GgufModel::open(path);auto e=ds41_read_geometry(m,g);require(e.empty(),e);require(g.engram_pad==1,"pad id must be compressed");require(g.cand_layer==-1,"negative candidate id");require(g.engram_rows==std::vector<int64_t>({5,7}),"layer row counts");e=ds41_check_tensors(m,g,{0,1});require(e.empty(),e);require(g.experts[0].gu_type[0]==16&&g.experts[0].d_type[0]==39,"mixed IQ/MXFP4 roles");}
    {
        auto optional = fixture();
        for (auto it = optional.keys.begin(); it != optional.keys.end(); ) {
            if (it->n == "deepseek41.engram.compressed_vocab_size" || it->n == "deepseek41.ple.head_offsets") it = optional.keys.erase(it);
            else ++it;
        }
        for (size_t i = optional.ts.size(); i-- > 0; ) {
            if (optional.ts[i].name.find("exp_probs_b_vl.bias") != std::string::npos) {
                optional.ts.erase(optional.ts.begin() + i); optional.payloads.erase(optional.payloads.begin() + i);
            }
        }
        optional.write(path); auto model = GgufModel::open(path); Ds41Geometry geometry;
        auto error = ds41_read_geometry(model, geometry); require(error.empty(), error);
        require(geometry.engram_cvocab == 4, "optional compressed vocabulary not inferred");
        error = ds41_check_tensors(model, geometry, {0,1}); require(error.empty(), error);
    }
    // Both exact grouped-output layouts are supported, not arbitrary equal products.
    {
        auto grouped = fixture();
        for (auto & k : grouped.keys) if (k.n == "deepseek41.attention.output_group_count") {
            k.value.b.clear(); k.value.put<int64_t>(2);
        }
        for (auto & t : grouped.ts) {
            if (t.name.find("attn_output_a.weight") != std::string::npos) t.shape = {2,4,2};
            if (t.name.find("attn_output_b.weight") != std::string::npos) t.shape = {8,256};
        }
        for (size_t i=0;i<grouped.ts.size();++i)
            grouped.payloads[i].resize((size_t)tensor_payload_bytes(grouped.ts[i]));
        auto check_map = [&](Fixture sample, bool accepted, const std::string & why) {
            sample.write(path); auto model = GgufModel::open(path); Ds41Geometry geom;
            auto error = ds41_read_geometry(model, geom); require(error.empty(), error);
            error = ds41_check_tensors(model, geom, {0,1});
            require(error.empty() == accepted, why + ": " + error);
        };
        check_map(grouped, true, "exact grouped 3-D view rejected");
        auto flatten = grouped;
        for (auto & t : flatten.ts) if (t.name.find("attn_output_a.weight") != std::string::npos) t.shape = {2,8};
        check_map(flatten, true, "exact flattened 2-D view rejected");
        auto half = flatten;
        for (size_t i=0;i<half.ts.size();++i) if (half.ts[i].name.find("attn_output_a.weight") != std::string::npos) {
            half.ts[i].type=1;half.payloads[i].resize((size_t)tensor_payload_bytes(half.ts[i]));
        }
        check_map(half, true, "existing llama.cpp F16 type policy changed");
        for (auto shape : std::vector<std::vector<uint64_t>>{{4,4},{8,2},{2,4,2,1},{2,2,4},{2,7}}) {
            auto bad = flatten;
            for (auto & t : bad.ts) if (t.name == "blk.0.attn_output_a.weight") t.shape = shape;
            check_map(bad, false, "wrong grouped-output shape accepted");
        }
    }
#ifdef _WIN32
    _putenv_s("STRATA_ENGRAM_CACHE_ROWS", "128"); _putenv_s("STRATA_ENGRAM_RAWDEV", "0");
#else
    setenv("STRATA_ENGRAM_CACHE_ROWS", "128", 1); setenv("STRATA_ENGRAM_RAWDEV", "0", 1);
#endif
    {Ds41Engram e;std::string err;require(e.open(g,{path},2,err),err);for(int i=0;i<9;++i)e.set_token(i,i%4);e.set_dead(3);
        std::vector<std::vector<float>> fast,ref;require(e.rows(0,9,fast,err),err);require(e.rows_reference(0,9,ref,err),err);require(fast==ref,"fast/reference row parity");
        for(int slot=0;slot<2;++slot)for(int t=0;t<9;++t){int64_t id;e.hash(t,slot,&id);for(int j=0;j<256;++j)require(fast[slot][t*256+j]==expected((int)id+(slot?5:0),j),"Q5_K scalar/combined layer view");}
        std::vector<std::vector<float>> again;auto job=e.start(0,9,again,err);require(job&&e.wait(job,err),err);require(again==fast,"async/cached parity");}
    // The previous per-layer IQ4_NL table convention remains usable with the same lazy reader.
    {
        Fixture old;
        const int8_t values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        for (int slot=0;slot<2;++slot) {
            old.tensor("blk."+std::to_string(slot)+".engram_embd.weight",{256,(uint64_t)(slot?7:5)},20);
            auto & bytes=old.payloads.back();
            for (size_t at=0;at<bytes.size();at+=18) { bytes[at+1]=0x38;for(int j=0;j<16;++j)bytes[at+2+j]=(uint8_t)(j|((15-j)<<4)); }
        }
        old.write(path);Ds41Engram reader;std::string err;require(reader.open(g,{path},1,err),err);
        reader.set_token(0,0);std::vector<std::vector<float>> rows;require(reader.rows(0,1,rows,err),err);
        for (int slot=0;slot<2;++slot)for(int j=0;j<256;++j)
            require(rows[slot][j]==.5f*values[j%32<16?j%16:15-j%16],"legacy IQ4_NL row changed");
    }
    auto reject=[&](Fixture bad,const std::string & what){bad.write(path);auto m=GgufModel::open(path);Ds41Geometry x;require(!ds41_read_geometry(m,x).empty(),what);};
    {auto bad=f;bad.set_array("deepseek41.ple.head_offsets",{0,6});reject(bad,"offset gap accepted");}
    {auto bad=f;bad.set_array("deepseek41.engram.token_map",{0,-1,1,2});reject(bad,"negative compressed id accepted");}
    {auto bad=f;bad.set_array("deepseek41.ple.layer_multipliers",{INT64_MAX,5,7,11});reject(bad,"overflow multiplier accepted");}
    {auto bad=f;bad.set_array("deepseek41.swiglu_clamp_shexp",{9,10});reject(bad,"shared clamp mismatch accepted");}
    {auto bad=f;bad.set_array("deepseek41.ple.layers",{1,0});reject(bad,"unordered layers accepted");}
    {auto bad=f;for(auto &k:bad.keys)if(k.n=="deepseek41.ple.heads_per_ngram"){k.value.b.clear();k.value.put<int64_t>(INT64_MAX);}reject(bad,"overflow head count accepted");}
    f.write(path);std::filesystem::resize_file(path,std::filesystem::file_size(path)-32);{Ds41Engram e;std::string err;require(!e.open(g,{path},1,err),"truncated combined tensor accepted");}
    std::cout<<"modern PLE geometry, full tensor map, mixed experts, combined Q5_K/dead-history/cache/async and malformed-input gates passed\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
