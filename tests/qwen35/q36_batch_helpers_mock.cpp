#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>
struct Mat {int type,n_in,n_out;const void* d;};
struct Context {int width=0,outwidth=0,rows=0;const char* input=nullptr;float* output=nullptr;const float* g=nullptr;const float* u=nullptr;char* dst=nullptr;std::vector<int> seen;int calls=0;} ctx;
namespace K {
size_t native_q8_1_bytes(int width,int rows){assert(width%32==0);return size_t(width/32)*36*rows;}
void count_ok(int n){if(n<1 || n>8)throw std::runtime_error("backend accepts only1..8 rows");}
void native_mmvq(int,const void*,const void* input,float* output,int width,int outwidth,int count,int){
 count_ok(count);assert(width==ctx.width && outwidth==ctx.outwidth);const size_t bytes=native_q8_1_bytes(width,1);
 ptrdiff_t offset=(const char*)input-ctx.input;assert(offset>=0 && size_t(offset)%bytes==0);int first=offset/bytes;
 assert(output-ctx.output==ptrdiff_t(first)*outwidth);assert(first+count<=ctx.rows);++ctx.calls;
 for(int row=first;row<first+count;++row){assert(ctx.seen[row]++==0);for(int col=0;col<outwidth;++col)output[(row-first)*outwidth+col]=float(row*1000+col);}
}
void native_swiglu_quantize_q8_1(const float* g,const float* u,void* dst,int width,int count,int){
 count_ok(count);assert(width==ctx.width);ptrdiff_t offset=g-ctx.g;assert(offset>=0 && offset%width==0);int first=offset/width;
 assert(u-ctx.u==ptrdiff_t(first)*width);const size_t bytes=native_q8_1_bytes(width,1);
 assert((char*)dst-ctx.dst==ptrdiff_t(first*bytes));assert(first+count<=ctx.rows);++ctx.calls;
 for(int row=first;row<first+count;++row){assert(ctx.seen[row]++==0);for(size_t col=0;col<bytes;++col)((unsigned char*)dst)[(row-first)*bytes+col]=(row*31+col)%251;}
}
}
struct Engine {static constexpr int kGemmT=8;int st=7;
#include "q36_batch_helpers_extracted.hpp"
};
int main(){
 Engine e;int cases=0;
 for(int width:{512,2048,4096})for(int rows:{1,4,8,9,16,32}){
  const size_t bytes=K::native_q8_1_bytes(width,1);int outwidth=width/2+7;
  std::vector<char> input((rows+2)*bytes,char(0x6a));std::vector<float> output((rows+2)*outwidth,-777.f);
  ctx={width,outwidth,rows,input.data()+bytes,output.data()+outwidth,nullptr,nullptr,nullptr,std::vector<int>(rows),0};
  Mat w{0,width,outwidth,nullptr};e.batch_apply(w,ctx.input,ctx.output,rows);
  assert(ctx.calls==(rows+7)/8);assert(std::all_of(ctx.seen.begin(),ctx.seen.end(),[](int n){return n==1;}));
  for(int row=0;row<rows;++row)for(int col=0;col<outwidth;++col)assert(output[(row+1)*outwidth+col]==float(row*1000+col));
  for(int col=0;col<outwidth;++col){assert(output[col]==-777.f);assert(output[(rows+1)*outwidth+col]==-777.f);}
  assert(std::all_of(input.begin(),input.end(),[](char x){return x==char(0x6a);}));++cases;
  std::vector<float> g((rows+2)*width,2.f),u((rows+2)*width,3.f);std::vector<char> dst((rows+2)*bytes,char(0x7f));
  ctx={width,0,rows,nullptr,nullptr,g.data()+width,u.data()+width,dst.data()+bytes,std::vector<int>(rows),0};
  e.batch_swiglu_quantize(ctx.g,ctx.u,ctx.dst,width,rows);
  assert(ctx.calls==(rows+7)/8);assert(std::all_of(ctx.seen.begin(),ctx.seen.end(),[](int n){return n==1;}));
  for(int row=0;row<rows;++row)for(size_t col=0;col<bytes;++col)assert((unsigned char)dst[(row+1)*bytes+col]==(row*31+col)%251);
  for(size_t col=0;col<bytes;++col){assert(dst[col]==char(0x7f));assert(dst[(rows+1)*bytes+col]==char(0x7f));}
  assert(std::all_of(g.begin(),g.end(),[](float x){return x==2.f;}));assert(std::all_of(u.begin(),u.end(),[](float x){return x==3.f;}));++cases;
 }
 bool rejected=false;try{K::count_ok(9);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 std::printf("PASS %d cases: exact helpers,rows1/4/8/9/16/32,widths512/2048/4096,count<=8,all-row offsets+coverage+sentinels;negativecount9rejected\n",cases);
}
