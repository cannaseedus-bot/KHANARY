// gpt2_block_run.cpp — run ONE full GPT-2 transformer block on the HD 4600 by CHAINING the
// verified glyph kernels on the GPU (ln1 -> qkv -> attn -> proj -> +res -> ln2 -> fc -> gelu ->
// proj -> +res), staying in GPU buffers between ops. Verify vs the CPU inference driver's block.
// This is the GPU port of the driver's graph walk (the full model = this block x12 + embed + lm_head).
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

static ID3D11Device* dev; static ID3D11DeviceContext* ctx;
static std::string rd(const std::string&p){std::ifstream f(p,std::ios::binary);std::stringstream s;s<<f.rdbuf();return s.str();}
static std::vector<float> rf(const std::string&p,size_t n){std::vector<float> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*4);return v;}

struct Buf { ID3D11Buffer* b=nullptr; ID3D11ShaderResourceView* srv=nullptr; ID3D11UnorderedAccessView* uav=nullptr; };
static Buf mkbuf(UINT count, const void* data, bool uav){
    Buf B; D3D11_BUFFER_DESC bd{}; bd.ByteWidth=count*4; bd.Usage=D3D11_USAGE_DEFAULT;
    bd.BindFlags=D3D11_BIND_SHADER_RESOURCE|(uav?D3D11_BIND_UNORDERED_ACCESS:0);
    bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; bd.StructureByteStride=4;
    D3D11_SUBRESOURCE_DATA sd{}; sd.pSysMem=data; dev->CreateBuffer(&bd, data?&sd:nullptr, &B.b);
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{}; sv.Format=DXGI_FORMAT_UNKNOWN; sv.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; sv.Buffer.NumElements=count; dev->CreateShaderResourceView(B.b,&sv,&B.srv);
    if(uav){ D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.Format=DXGI_FORMAT_UNKNOWN; ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; ud.Buffer.NumElements=count; dev->CreateUnorderedAccessView(B.b,&ud,&B.uav); }
    return B;
}
static ID3D11Buffer* cbuf(const void*d,UINT bytes){D3D11_BUFFER_DESC bd{};bd.ByteWidth=(bytes+15)&~15u;bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA sd{};sd.pSysMem=d;ID3D11Buffer*b=nullptr;dev->CreateBuffer(&bd,&sd,&b);return b;}
static ID3D11ComputeShader* csFromSrc(const std::string&hlsl){ID3DBlob*bl=nullptr,*er=nullptr;if(FAILED(D3DCompile(hlsl.data(),hlsl.size(),"k",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&bl,&er))){if(er)printf("HLSL err: %s\n",(char*)er->GetBufferPointer());return nullptr;}ID3D11ComputeShader*cs=nullptr;dev->CreateComputeShader(bl->GetBufferPointer(),bl->GetBufferSize(),nullptr,&cs);return cs;}

static void run(ID3D11ComputeShader*cs, std::vector<ID3D11ShaderResourceView*> srvs, std::vector<ID3D11UnorderedAccessView*> uavs, ID3D11Buffer*cb, UINT gx,UINT gy){
    // clear stale bindings so a buffer never lingers as UAV while read as SRV (D3D11 hazard)
    ID3D11ShaderResourceView* nsrv[4]={nullptr,nullptr,nullptr,nullptr};
    ID3D11UnorderedAccessView* nuav[3]={nullptr,nullptr,nullptr};
    ctx->CSSetShaderResources(0,4,nsrv);
    ctx->CSSetUnorderedAccessViews(0,3,nuav,nullptr);
    ctx->CSSetShader(cs,nullptr,0);
    if(!srvs.empty()) ctx->CSSetShaderResources(0,(UINT)srvs.size(),srvs.data());
    if(!uavs.empty()) ctx->CSSetUnorderedAccessViews(0,(UINT)uavs.size(),uavs.data(),nullptr);
    ctx->CSSetConstantBuffers(0,1,&cb);
    ctx->Dispatch(gx,gy,1);
}
struct M4{uint32_t a,b,c,d;};

int main(){
    UINT S,E,NH; { std::stringstream d(rd("dims.txt")); d>>S>>E>>NH; }
    UINT E3=3*E, E4=4*E, D=E/NH;
    D3D_FEATURE_LEVEL fl{}; D3D_FEATURE_LEVEL want[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
    if(FAILED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,want,2,D3D11_SDK_VERSION,&dev,&fl,&ctx))){printf("no d3d11\n");return 1;}
    printf("[dev] D3D11 FL 0x%x  chaining 1 gpt2 block on GPU: S=%u E=%u NH=%u\n",(int)fl,S,E,NH);

    auto csLN=csFromSrc(rd("k_layernorm.hlsl")), csMM=csFromSrc(rd("k_matmul.hlsl")),
         csAT=csFromSrc(rd("k_attention.hlsl")), csGE=csFromSrc(rd("k_gelu.hlsl")),
         csBI=csFromSrc(rd("k_add_bias.hlsl")), csRE=csFromSrc(rd("k_add.hlsl"));   // G_ADD_BIAS / G_ADD glyphs
    if(!csLN||!csMM||!csAT||!csGE||!csBI||!csRE){printf("kernel compile failed\n");return 1;}

    // weights (SRV-only)
    auto Wb=[&](const char*n,UINT c){ return mkbuf(c, rf(std::string(n)+".bin",c).data(), false); };
    Buf x   = mkbuf(S*E, rf("x.bin",S*E).data(), true);          // r1 (also SRV)
    Buf ln1g=Wb("ln_1_weight",E), ln1b=Wb("ln_1_bias",E);
    Buf caw =Wb("attn_c_attn_weight",E*E3), cab=Wb("attn_c_attn_bias",E3);
    Buf cpw =Wb("attn_c_proj_weight",E*E), cpb=Wb("attn_c_proj_bias",E);
    Buf ln2g=Wb("ln_2_weight",E), ln2b=Wb("ln_2_bias",E);
    Buf fcw =Wb("mlp_c_fc_weight",E*E4), fcb=Wb("mlp_c_fc_bias",E4);
    Buf ppw =Wb("mlp_c_proj_weight",E4*E), ppb=Wb("mlp_c_proj_bias",E);
    // activations (SRV+UAV)
    Buf h1=mkbuf(S*E,nullptr,true), xh=mkbuf(S*E,nullptr,true), is=mkbuf(S,nullptr,true);
    Buf qkv=mkbuf(S*E3,nullptr,true), att=mkbuf(S*E,nullptr,true), pbuf=mkbuf(NH*S*S,nullptr,true);
    Buf ap=mkbuf(S*E,nullptr,true), x2=mkbuf(S*E,nullptr,true);
    Buf h2=mkbuf(S*E,nullptr,true), xh2=mkbuf(S*E,nullptr,true), is2=mkbuf(S,nullptr,true);
    Buf fc=mkbuf(S*E4,nullptr,true), fcg=mkbuf(S*E4,nullptr,true), fp=mkbuf(S*E,nullptr,true);

    struct LN{uint32_t n_embd,seq_len;float eps;uint32_t pad;};
    struct MM{uint32_t M,N,K,pad;}; struct AT{uint32_t S,E,D;float sc;};
    struct GE{uint32_t numel,off,p0,p1;}; struct BI{uint32_t rows,N,p0,p1;}; struct RE{uint32_t len,p0,p1,p2;};
    auto D1=[&](UINT n){return (n+255)/256;};

    // ln1: x -> h1
    { LN c{E,S,1e-5f,0}; run(csLN,{x.srv,ln1g.srv,ln1b.srv},{h1.uav,xh.uav,is.uav},cbuf(&c,sizeof c),S,1); }
    // qkv = h1 @ caw ; += cab
    { MM c{S,E3,E,0}; run(csMM,{h1.srv,caw.srv},{qkv.uav},cbuf(&c,sizeof c),(E3+15)/16,(S+15)/16); }
    { BI c{S,E3,0,0}; run(csBI,{cab.srv},{qkv.uav},cbuf(&c,sizeof c),D1(S*E3),1); }
    // att = attention(qkv)
    { AT c{S,E,D,1.0f/sqrtf((float)D)}; run(csAT,{qkv.srv},{att.uav,pbuf.uav},cbuf(&c,sizeof c),NH,1); }
    // ap = att @ cpw ; += cpb ; x2 = ap + r1(x)
    { MM c{S,E,E,0}; run(csMM,{att.srv,cpw.srv},{ap.uav},cbuf(&c,sizeof c),(E+15)/16,(S+15)/16); }
    { BI c{S,E,0,0}; run(csBI,{cpb.srv},{ap.uav},cbuf(&c,sizeof c),D1(S*E),1); }
    ctx->CopyResource(x2.b,ap.b);
    { RE c{S*E,0,0,0}; run(csRE,{x.srv},{x2.uav},cbuf(&c,sizeof c),D1(S*E),1); }
    // ln2: x2 -> h2   (r2 = x2)
    { LN c{E,S,1e-5f,0}; run(csLN,{x2.srv,ln2g.srv,ln2b.srv},{h2.uav,xh2.uav,is2.uav},cbuf(&c,sizeof c),S,1); }
    // fc = h2 @ fcw ; += fcb ; gelu(fc)
    { MM c{S,E4,E,0}; run(csMM,{h2.srv,fcw.srv},{fc.uav},cbuf(&c,sizeof c),(E4+15)/16,(S+15)/16); }
    { BI c{S,E4,0,0}; run(csBI,{fcb.srv},{fc.uav},cbuf(&c,sizeof c),D1(S*E4),1); }
    { GE c{S*E4,0,0,0}; run(csGE,{fc.srv},{fcg.uav},cbuf(&c,sizeof c),D1(S*E4),1); }   // separate out buf (no SRV/UAV alias)
    // fp = fcg @ ppw ; += ppb ; out = fp + r2(x2)
    { MM c{S,E,E4,0}; run(csMM,{fcg.srv,ppw.srv},{fp.uav},cbuf(&c,sizeof c),(E+15)/16,(S+15)/16); }
    { BI c{S,E,0,0}; run(csBI,{ppb.srv},{fp.uav},cbuf(&c,sizeof c),D1(S*E),1); }
    { RE c{S*E,0,0,0}; run(csRE,{x2.srv},{fp.uav},cbuf(&c,sizeof c),D1(S*E),1); }

    // readback fp
    std::vector<float> out(S*E), ref=rf("ref.bin",S*E);
    D3D11_BUFFER_DESC st{}; st.ByteWidth=S*E*4; st.Usage=D3D11_USAGE_STAGING; st.CPUAccessFlags=D3D11_CPU_ACCESS_READ; ID3D11Buffer*stg=nullptr; dev->CreateBuffer(&st,nullptr,&stg);
    ctx->CopyResource(stg,fp.b); D3D11_MAPPED_SUBRESOURCE ms; ctx->Map(stg,0,D3D11_MAP_READ,0,&ms); memcpy(out.data(),ms.pData,S*E*4); ctx->Unmap(stg,0);

    double mx=0,am=0; for(size_t i=0;i<out.size();i++){double a=std::fabs((double)out[i]-ref[i]); if(a>mx)mx=a; if(std::fabs((double)ref[i])>am)am=std::fabs((double)ref[i]);}
    printf("[out 0..2]=(%.5f,%.5f,%.5f) ref=(%.5f,%.5f,%.5f)\n",out[0],out[1],out[2],ref[0],ref[1],ref[2]);
    printf("[verify] max abs err=%.3e  scale=%.3f  scale-norm=%.2e\n\n",mx,am,mx/am);
    bool pass=(mx/am)<1e-4;
    printf("=== %s: GPU-chained gpt2 block on HD4600 -> %s vs CPU inference driver ===\n",pass?"PASS":"FAIL",pass?"matches":"MISMATCH");
    return pass?0:1;
}
