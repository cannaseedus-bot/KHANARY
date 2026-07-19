#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#include <d3d11.h>
#include <wrl/client.h>

#include "xvm_core.h"

using Microsoft::WRL::ComPtr;

class D3D11Engine {
 public:
  bool init(bool forceWarp = false, bool verboseLog = false);
  bool uploadVM(XVMState& vm);
  bool dispatch(XVMState& vm);
  bool dumpTrace(std::uint32_t maxEntries = 32);
  bool streamSCXQ2ToCodeBuffer(const wchar_t*, std::uint64_t) { return false; }
  bool streamFileSliceToCodeBuffer(const wchar_t*, std::uint64_t, std::uint64_t) { return false; }

  bool usedWarp() const { return usedWarp_; }
  D3D_FEATURE_LEVEL featureLevel() const { return featureLevel_; }
  const std::string& adapterName() const { return adapterName_; }
  const std::string& initReason() const { return initReason_; }

  // Trainer layer access
  ID3D11Device*        rawDevice() const { return device_.Get(); }
  ID3D11DeviceContext* rawCtx()    const { return ctx_.Get(); }

 private:
  ComPtr<ID3D11Device> device_;
  ComPtr<ID3D11DeviceContext> ctx_;
  ComPtr<ID3D11ComputeShader> shader_;
  bool usedWarp_ = false;
  D3D_FEATURE_LEVEL featureLevel_ = D3D_FEATURE_LEVEL_9_1;
  std::string adapterName_;
  std::string initReason_;

  ComPtr<ID3D11Buffer> code_;
  ComPtr<ID3D11Buffer> fibers_;
  ComPtr<ID3D11Buffer> shared_;
  ComPtr<ID3D11Buffer> stack_;
  ComPtr<ID3D11Buffer> trace_;
  ComPtr<ID3D11Buffer> traceIndex_;

  ComPtr<ID3D11ShaderResourceView> codeSrv_;
  ComPtr<ID3D11UnorderedAccessView> fibersUav_;
  ComPtr<ID3D11UnorderedAccessView> sharedUav_;
  ComPtr<ID3D11UnorderedAccessView> stackUav_;
  ComPtr<ID3D11UnorderedAccessView> traceUav_;
  ComPtr<ID3D11UnorderedAccessView> traceIndexUav_;

  ComPtr<ID3D11Buffer> traceReadback_;
  ComPtr<ID3D11Buffer> traceIndexReadback_;

  std::uint32_t traceCapacityWords_ = 131072;

  bool loadShader();
  bool createBuffers(XVMState& vm);
  bool createStaging(std::uint32_t traceWords);
};
