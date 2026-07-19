// gpt2_train_main.cpp — entry point for D3D11 GPU trainer
//
// Usage:
//   gpt2_trainer --model models/cli_coder_gpt2/model.safetensors
//                --data  train/out/tokens.bin
//                --out   models/cli_coder_gpt2_dx11/model.safetensors
//                --steps 1000 --batch 4 --block 128 --lr 3e-5

#include "gpt2_trainer.h"
#include "d3d11_engine.h"   // resolved via -I xvm-d3d12/src (matches gpt2_trainer.h)
#include <iostream>
#include <string>
#include <cstring>

static std::string argval(int argc, char** argv, const char* key, const char* def = "") {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i+1];
    return def;
}

int main(int argc, char** argv) {
    TrainerConfig cfg;
    cfg.model_path   = argval(argc, argv, "--model",
                              "C:/Users/canna/.micronaut/models/cli_coder_gpt2/model.safetensors");
    cfg.data_path    = argval(argc, argv, "--data",
                              "C:/Users/canna/.micronaut/train/out/tokens.bin");
    cfg.output_path  = argval(argc, argv, "--out",
                              "C:/Users/canna/.micronaut/models/cli_coder_dx11/model.safetensors");
    cfg.max_steps    = std::stoi(argval(argc, argv, "--steps", "1000"));
    cfg.batch_size   = std::stoi(argval(argc, argv, "--batch", "4"));
    cfg.block_size   = std::stoi(argval(argc, argv, "--block", "128"));
    cfg.lr           = std::stof(argval(argc, argv, "--lr", "3e-5"));
    cfg.save_every   = std::stoi(argval(argc, argv, "--save-every", "200"));
    cfg.use_gpu_fwd  = true;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--no-gpu-fwd") == 0) { cfg.use_gpu_fwd = false; break; }

    // Init D3D11
    D3D11Engine d11;
    if (!d11.init(/*forceWarp=*/false, /*verbose=*/true)) {
        std::cerr << "[main] D3D11 init failed: " << d11.initReason() << "\n";
        return 1;
    }
    std::cerr << "[main] D3D11 adapter: " << d11.adapterName()
              << " (feature level " << (int)d11.featureLevel() << ")\n";

    GPT2Trainer trainer(&d11);
    if (!trainer.init(cfg)) {
        std::cerr << "[main] trainer init failed\n";
        return 1;
    }

    std::cerr << "[main] starting training: " << cfg.max_steps << " steps\n";
    trainer.train();

    std::cerr << "[main] done\n";
    return 0;
}
