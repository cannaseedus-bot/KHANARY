#pragma once

#include <cstdint>
#include <thread>
#include <vector>

struct XVMFiber {
  std::uint32_t pc = 0;
  std::uint32_t sp = 0;
  std::uint32_t phase = 0;
  std::uint32_t flags = 1;
  std::uint32_t r0 = 0;
  std::uint32_t r1 = 0;
  std::uint32_t r2 = 0;
  std::uint32_t r3 = 0;
};

struct XVMState {
  std::vector<std::uint8_t> code;
  std::vector<std::uint8_t> constPool;
  std::vector<XVMFiber> fibers;
  std::vector<std::uint32_t> shared;
  std::uint64_t tick = 0;

  std::uint32_t fiberCount() const {
    return static_cast<std::uint32_t>(fibers.size());
  }
};

void xvm_run_cpu_step(XVMState& vm, std::uint32_t fid);
void xvm_run_cpu_ticks(XVMState& vm, std::uint64_t ticks);
void xvm_run_cpu_ticks_mt(XVMState& vm, std::uint64_t ticks, std::uint32_t threadCount = std::thread::hardware_concurrency());
