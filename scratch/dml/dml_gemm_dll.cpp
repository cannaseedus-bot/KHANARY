// dml_gemm_dll.cpp — DirectML GEMM as a C-ABI DLL, for wiring the DML matmul path into the
// KHANARY inference driver (tools/kxml_inference_driver.py) via ctypes.
//
// Exports: int dml_gemm_f32(const float* A, const float* B, float* C, uint M, uint N, uint K)
//   computes C[M,N] = A[M,K] @ B[K,N] (row-major) on DirectML (D3D12, HD 4600 FL 11_1).
// Persistent D3D12+DML device; compiled GEMM operators cached by (M,N,K) shape.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <DirectML.h>
#include <map>
#include <tuple>
#include <cstdint>
#include <algorithm>
#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"DirectML.lib")

#define OK(x) do{ if(FAILED(x)) return -1; }while(0)

static D3D12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE t){ D3D12_HEAP_PROPERTIES h{}; h.Type=t; h.CPUPageProperty=D3D12_CPU_PAGE_PROPERTY_UNKNOWN; h.MemoryPoolPreference=D3D12_MEMORY_POOL_UNKNOWN; h.CreationNodeMask=1; h.VisibleNodeMask=1; return h; }
static D3D12_RESOURCE_DESC bufDesc(UINT64 b,D3D12_RESOURCE_FLAGS f=D3D12_RESOURCE_FLAG_NONE){ D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=b; d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=f; return d; }
static D3D12_RESOURCE_BARRIER barrier(ID3D12Resource*r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){ D3D12_RESOURCE_BARRIER br{}; br.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br.Transition.pResource=r; br.Transition.StateBefore=a; br.Transition.StateAfter=b; br.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; return br; }

struct Ctx {
    ID3D12Device* dev=nullptr; ID3D12CommandQueue* q=nullptr; ID3D12CommandAllocator* alloc=nullptr;
    ID3D12GraphicsCommandList* cl=nullptr; ID3D12Fence* fence=nullptr; UINT64 fv=0; HANDLE fe=nullptr;
    IDMLDevice* dml=nullptr; IDMLCommandRecorder* rec=nullptr;
    bool init=false;
    struct Op { IDMLCompiledOperator* op; ID3D12DescriptorHeap* heap; IDMLBindingTable* bt; UINT descN; };
    std::map<std::tuple<UINT,UINT,UINT>,Op> cache;
    void flush(){ cl->Close(); ID3D12CommandList*ls[]={cl}; q->ExecuteCommandLists(1,ls); q->Signal(fence,++fv); fence->SetEventOnCompletion(fv,fe); WaitForSingleObject(fe,INFINITE); alloc->Reset(); cl->Reset(alloc,nullptr); }
};
static Ctx g;

static int ensureInit(){
    if(g.init) return 0;
    IDXGIFactory4* fac=nullptr; OK(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
    IDXGIAdapter1* ad=nullptr; DXGI_ADAPTER_DESC1 dd{};
    for(UINT i=0; fac->EnumAdapters1(i,&ad)!=DXGI_ERROR_NOT_FOUND; ++i){
        ad->GetDesc1(&dd); if(dd.Flags & DXGI_ADAPTER_FLAG_SOFTWARE){ ad->Release(); ad=nullptr; continue; }
        if(SUCCEEDED(D3D12CreateDevice(ad,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&g.dev)))) break;
        ad->Release(); ad=nullptr;
    }
    if(!g.dev) return -1;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    OK(g.dev->CreateCommandQueue(&qd,IID_PPV_ARGS(&g.q)));
    OK(g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&g.alloc)));
    OK(g.dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,g.alloc,nullptr,IID_PPV_ARGS(&g.cl)));
    OK(g.dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&g.fence)));
    g.fe=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    OK(DMLCreateDevice(g.dev,DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&g.dml)));
    OK(g.dml->CreateCommandRecorder(IID_PPV_ARGS(&g.rec)));
    g.init=true; return 0;
}

static UINT64 calcSize(UINT a,UINT b){ return (UINT64)a*b*sizeof(float); }

static int getOp(UINT M,UINT N,UINT K,Ctx::Op& out){
    auto key=std::make_tuple(M,N,K);
    auto it=g.cache.find(key);
    if(it!=g.cache.end()){ out=it->second; return 0; }
    UINT aSz[4]={1,1,M,K}, bSz[4]={1,1,K,N}, oSz[4]={1,1,M,N};
    auto mk=[&](UINT*s,DML_BUFFER_TENSOR_DESC&bt,DML_TENSOR_DESC&td){ bt={}; bt.DataType=DML_TENSOR_DATA_TYPE_FLOAT32; bt.DimensionCount=4; bt.Sizes=s; bt.TotalTensorSizeInBytes=(UINT64)s[0]*s[1]*s[2]*s[3]*sizeof(float); td.Type=DML_TENSOR_TYPE_BUFFER; td.Desc=&bt; };
    DML_BUFFER_TENSOR_DESC aBT,bBT,oBT; DML_TENSOR_DESC aTD,bTD,oTD; mk(aSz,aBT,aTD); mk(bSz,bBT,bTD); mk(oSz,oBT,oTD);
    DML_GEMM_OPERATOR_DESC gemm{}; gemm.ATensor=&aTD; gemm.BTensor=&bTD; gemm.CTensor=nullptr; gemm.OutputTensor=&oTD;
    gemm.TransA=DML_MATRIX_TRANSFORM_NONE; gemm.TransB=DML_MATRIX_TRANSFORM_NONE; gemm.Alpha=1.0f; gemm.Beta=0.0f;
    DML_OPERATOR_DESC opd{}; opd.Type=DML_OPERATOR_GEMM; opd.Desc=&gemm;
    IDMLOperator* op=nullptr; OK(g.dml->CreateOperator(&opd,IID_PPV_ARGS(&op)));
    Ctx::Op o{}; OK(g.dml->CompileOperator(op,DML_EXECUTION_FLAG_NONE,IID_PPV_ARGS(&o.op))); op->Release();
    IDMLOperatorInitializer* init=nullptr; IDMLCompiledOperator* cops[]={o.op};
    OK(g.dml->CreateOperatorInitializer(1,cops,IID_PPV_ARGS(&init)));
    DML_BINDING_PROPERTIES ip=init->GetBindingProperties(), ep=o.op->GetBindingProperties();
    o.descN=std::max(ip.RequiredDescriptorCount,ep.RequiredDescriptorCount);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=o.descN; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    OK(g.dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&o.heap)));
    DML_BINDING_TABLE_DESC btd{}; btd.Dispatchable=init; btd.CPUDescriptorHandle=o.heap->GetCPUDescriptorHandleForHeapStart(); btd.GPUDescriptorHandle=o.heap->GetGPUDescriptorHandleForHeapStart(); btd.SizeInDescriptors=o.descN;
    OK(g.dml->CreateBindingTable(&btd,IID_PPV_ARGS(&o.bt)));
    // initialize (GEMM has no persistent resource; temp for init is usually 0)
    ID3D12DescriptorHeap* heaps[]={o.heap}; g.cl->SetDescriptorHeaps(1,heaps);
    g.rec->RecordDispatch(g.cl,init,o.bt); g.flush(); init->Release();
    g.cache[key]=o; out=o; return 0;
}

extern "C" __declspec(dllexport) int dml_gemm_f32(const float* A,const float* B,float* C,unsigned M,unsigned N,unsigned K){
    if(ensureInit()) return -2;
    Ctx::Op o; if(getOp(M,N,K,o)) return -3;
    UINT64 aB=calcSize(M,K), bB=calcSize(K,N), oB=calcSize(M,N);
    auto mkDefault=[&](UINT64 b)->ID3D12Resource*{ ID3D12Resource*r=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_DEFAULT); auto rd=bufDesc(b,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS); g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&r)); return r; };
    auto upload=[&](const float* data,UINT64 bytes)->ID3D12Resource*{
        ID3D12Resource* gpu=mkDefault(bytes); ID3D12Resource* up=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_UPLOAD); auto rd=bufDesc(bytes);
        g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&up));
        void* p=nullptr; D3D12_RANGE nr{0,0}; up->Map(0,&nr,&p); memcpy(p,data,bytes); up->Unmap(0,nullptr);
        auto b1=barrier(gpu,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST); g.cl->ResourceBarrier(1,&b1);
        g.cl->CopyBufferRegion(gpu,0,up,0,bytes);
        auto b2=barrier(gpu,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); g.cl->ResourceBarrier(1,&b2);
        g.flush(); up->Release(); return gpu; };
    ID3D12Resource* aBuf=upload(A,aB); ID3D12Resource* bBuf=upload(B,bB); ID3D12Resource* oBuf=mkDefault(oB);
    if(!aBuf||!bBuf||!oBuf) return -4;

    DML_BINDING_TABLE_DESC btd{}; btd.Dispatchable=o.op; btd.CPUDescriptorHandle=o.heap->GetCPUDescriptorHandleForHeapStart(); btd.GPUDescriptorHandle=o.heap->GetGPUDescriptorHandleForHeapStart(); btd.SizeInDescriptors=o.descN;
    o.bt->Reset(&btd);
    DML_BUFFER_BINDING inBB[2]={ {aBuf,0,aB},{bBuf,0,bB} };
    DML_BINDING_DESC inBD[3]={ {DML_BINDING_TYPE_BUFFER,&inBB[0]},{DML_BINDING_TYPE_BUFFER,&inBB[1]},{DML_BINDING_TYPE_NONE,nullptr} };
    o.bt->BindInputs(3,inBD);
    DML_BUFFER_BINDING oBB{oBuf,0,oB}; DML_BINDING_DESC oBD{DML_BINDING_TYPE_BUFFER,&oBB}; o.bt->BindOutputs(1,&oBD);
    ID3D12DescriptorHeap* heaps[]={o.heap}; g.cl->SetDescriptorHeaps(1,heaps);
    g.rec->RecordDispatch(g.cl,o.op,o.bt); g.flush();

    // readback
    ID3D12Resource* rb=nullptr; auto hp=heapProps(D3D12_HEAP_TYPE_READBACK); auto rd=bufDesc(oB);
    g.dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb));
    auto b1=barrier(oBuf,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); g.cl->ResourceBarrier(1,&b1);
    g.cl->CopyResource(rb,oBuf);
    g.flush();
    void* p=nullptr; D3D12_RANGE r{0,(SIZE_T)oB}; rb->Map(0,&r,&p); memcpy(C,p,oB); rb->Unmap(0,nullptr);
    rb->Release(); aBuf->Release(); bBuf->Release(); oBuf->Release();
    return 0;
}
