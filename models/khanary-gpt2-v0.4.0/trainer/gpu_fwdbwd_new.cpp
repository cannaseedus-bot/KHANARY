float GPT2Trainer::gpuForwardBackward(const std::vector<int32_t>& seq, float inv_batch) {
    auto* ctx = d11_->rawCtx();
    auto* dev = d11_->rawDevice();
    const uint32_t S  = (uint32_t)seq.size();
    const uint32_t E  = model_cfg_.n_embd;
    const uint32_t H  = model_cfg_.n_head;
    const uint32_t D  = model_cfg_.d_head;
    const uint32_t F  = model_cfg_.d_ff;
    const uint32_t V  = model_cfg_.vocab_size;
    const uint32_t NL = model_cfg_.n_layer;
    const float    eps   = 1e-5f;
    const float    scale = model_cfg_.attn_scale;

    // Helper: zero a structured buffer range via ClearUnorderedAccessViewUint
    auto zero = [&](ID3D11Buffer* b, uint32_t first, uint32_t n) {
        auto u = makeUAV(b, first, n);
        UINT z[4] = {};
        ctx->ClearUnorderedAccessViewUint(u.Get(), z);
    };

    // ── Upload tokens ─────────────────────────────────────────────────────────
    {
        D3D11_BUFFER_DESC sd{};
        sd.ByteWidth = S * 4; sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        ComPtr<ID3D11Buffer> st; dev->CreateBuffer(&sd, nullptr, &st);
        D3D11_MAPPED_SUBRESOURCE ms{};
        ctx->Map(st.Get(), 0, D3D11_MAP_WRITE, 0, &ms);
        std::memcpy(ms.pData, seq.data(), S * 4);
        ctx->Unmap(st.Get(), 0);
        ctx->CopyResource(tokens_buf_.Get(), st.Get());
    }

    // ══════════════════════════ FORWARD PASS ═════════════════════════════════

    // 1. Embed: wte[tokens] + wpe → h_buf_[0, S, E]
    {
        struct { uint32_t S, E, pad[2]; } p{S, E};
        setCB(&p, 16);
        ctx->CSSetShader(cs_embed_fwd_.Get(), nullptr, 0);
        auto tsrv = makeSRVi(tokens_buf_.Get(), 0, S);
        ID3D11ShaderResourceView* srvs[3] = {
            tsrv.Get(), wSRV("transformer.wte.weight"), wSRV("transformer.wpe.weight")
        };
        ctx->CSSetShaderResources(0, 3, srvs);
        zero(h_buf_.Get(), 0, S * E);
        auto uv0 = makeUAV(h_buf_.Get(), 0, S * E);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch(S, 1, 1); clearViews(1, 3);
    }

    // 2. Layer loop
    for (uint32_t l = 0; l < NL; ++l) {
        const std::string pfx = "transformer.h." + std::to_string(l) + ".";
        const uint32_t h_in  = l       * S * E;
        const uint32_t h_out = (l + 1) * S * E;
        const uint32_t qoff  = l * S * 3 * E;
        const uint32_t is1   = l * S;
        const uint32_t is2   = l * S;
        const uint32_t Poff  = l * H * S * S;
        const uint32_t loff1 = l * S * E;
        const uint32_t loff2 = l * S * E;
        const uint32_t aoff  = l * S * E;
        const uint32_t mpre  = l * S * F;
        const uint32_t mgel  = l * S * F;

        // LN1
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            ctx->CSSetShader(cs_lnorm_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(h_buf_.Get(), h_in, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "ln_1.weight"), wSRV(pfx + "ln_1.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            auto uv0 = makeUAV(ln1_y_buf_.Get(),    loff1, S * E);
            auto uv1 = makeUAV(xhat_ln1_buf_.Get(), loff1, S * E);
            auto uv2 = makeUAV(inv_std_ln1_.Get(),  is1,   S);
            ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);
        }

        // QKV projection
        {
            struct { uint32_t M, K, N, use_bias; } p{S, E, 3 * E, 1};
            setCB(&p, 16);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(ln1_y_buf_.Get(), loff1, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "attn.c_attn.weight"), wSRV(pfx + "attn.c_attn.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(qkv_buf_.Get(), qoff, S * 3 * E);
            auto uv0 = makeUAV(qkv_buf_.Get(), qoff, S * 3 * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((3 * E + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // Causal self-attention
        {
            struct { uint32_t S, E, D; float scale; } p{S, E, D, scale};
            setCB(&p, 16);
            ctx->CSSetShader(cs_attn_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(qkv_buf_.Get(), qoff, S * 3 * E);
            ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            zero(attn_out_buf_.Get(), aoff, S * E);
            auto uv0 = makeUAV(attn_out_buf_.Get(), aoff,  S * E);
            auto uv1 = makeUAV(P_buf_.Get(),         Poff, H * S * S);
            ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
            ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
            ctx->Dispatch(H, 1, 1); clearViews(2, 1);
        }

        // c_proj_attn → dh_buf_ (temp)
        {
            struct { uint32_t M, K, N, use_bias; } p{S, E, E, 1};
            setCB(&p, 16);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(attn_out_buf_.Get(), aoff, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "attn.c_proj.weight"), wSRV(pfx + "attn.c_proj.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(dh_buf_.Get(), 0, S * E);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((E + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // First residual: h_buf_[l+1] = h_buf_[l] + dh_buf_
        {
            struct { uint32_t numel, pad[3]; } p{S * E};
            setCB(&p, 16);
            ctx->CSSetShader(cs_resadd_add3_.Get(), nullptr, 0);
            auto sv0 = makeSRV(h_buf_.Get(),  h_in, S * E);
            auto sv1 = makeSRV(dh_buf_.Get(), 0,    S * E);
            ID3D11ShaderResourceView* srvs[2] = { sv0.Get(), sv1.Get() };
            ctx->CSSetShaderResources(0, 2, srvs);
            auto uv0 = makeUAV(h_buf_.Get(), h_out, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * E + 255) / 256, 1, 1); clearViews(1, 2);
        }

        // LN2
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            ctx->CSSetShader(cs_lnorm_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(h_buf_.Get(), h_out, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "ln_2.weight"), wSRV(pfx + "ln_2.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            auto uv0 = makeUAV(ln2_y_buf_.Get(),    loff2, S * E);
            auto uv1 = makeUAV(xhat_ln2_buf_.Get(), loff2, S * E);
            auto uv2 = makeUAV(inv_std_ln2_.Get(),  is2,   S);
            ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);
        }

        // c_fc → mlp_pre_buf_[l]
        {
            struct { uint32_t M, K, N, use_bias; } p{S, E, F, 1};
            setCB(&p, 16);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(ln2_y_buf_.Get(), loff2, S * E);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "mlp.c_fc.weight"), wSRV(pfx + "mlp.c_fc.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(mlp_pre_buf_.Get(), mpre, S * F);
            auto uv0 = makeUAV(mlp_pre_buf_.Get(), mpre, S * F);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((F + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // GELU: mlp_pre_buf_[l] → mlp_gelu_buf_[l]  (pre-activation stays in mlp_pre_buf_)
        // d_mlp_buf_ receives the x_pre copy (discarded)
        {
            struct { uint32_t numel, pad[3]; } p{S * F};
            setCB(&p, 16);
            ctx->CSSetShader(cs_gelu_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(mlp_pre_buf_.Get(), mpre, S * F);
            ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            auto uv0 = makeUAV(d_mlp_buf_.Get(),    0,    S * F);  // x_pre scratch
            auto uv1 = makeUAV(mlp_gelu_buf_.Get(), mgel, S * F);  // GELU output
            ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
            ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
            ctx->Dispatch((S * F + 255) / 256, 1, 1); clearViews(2, 1);
        }

        // c_proj_mlp → dh_buf_ (temp)
        {
            struct { uint32_t M, K, N, use_bias; } p{S, F, E, 1};
            setCB(&p, 16);
            ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(mlp_gelu_buf_.Get(), mgel, S * F);
            ID3D11ShaderResourceView* srvs[3] = {
                sv0.Get(), wSRV(pfx + "mlp.c_proj.weight"), wSRV(pfx + "mlp.c_proj.bias")
            };
            ctx->CSSetShaderResources(0, 3, srvs);
            zero(dh_buf_.Get(), 0, S * E);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((E + 15) / 16, (S + 15) / 16, 1); clearViews(1, 3);
        }

        // Second residual: h_buf_[l+1] += dh_buf_
        {
            struct { uint32_t numel, pad[3]; } p{S * E};
            setCB(&p, 16);
            ctx->CSSetShader(cs_resadd_addto_.Get(), nullptr, 0);
            auto sv0 = makeSRV(dh_buf_.Get(), 0, S * E);
            ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
            ctx->CSSetShaderResources(0, 1, srvs);
            auto uv0 = makeUAV(h_buf_.Get(), h_out, S * E);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * E + 255) / 256, 1, 1); clearViews(1, 1);
        }
    } // end layer loop

    // 3. Final LN: h_buf_[NL] → lnf_y_buf_, xhat_lnf, inv_std_lnf
    {
        const uint32_t h_last = NL * S * E;
        struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
        setCB(&p, 16);
        ctx->CSSetShader(cs_lnorm_fwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(h_buf_.Get(), h_last, S * E);
        ID3D11ShaderResourceView* srvs[3] = {
            sv0.Get(), wSRV("transformer.ln_f.weight"), wSRV("transformer.ln_f.bias")
        };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto uv0 = makeUAV(lnf_y_buf_.Get(),    0, S * E);
        auto uv1 = makeUAV(xhat_lnf_buf_.Get(), 0, S * E);
        auto uv2 = makeUAV(inv_std_lnf_.Get(),  0, S);
        ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
        ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
        ctx->Dispatch(S, 1, 1); clearViews(3, 3);
    }

    // 4. LM head: lnf_y_buf_[last, E] @ wte.T → logits_buf_[V]
    const uint32_t last = S - 1;
    {
        struct { uint32_t M, K, N, pad; } p{1, E, V, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_fwd_transb_.Get(), nullptr, 0);
        auto sv0 = makeSRV(lnf_y_buf_.Get(), last * E, E);
        ID3D11ShaderResourceView* srvs[2] = { sv0.Get(), wSRV("transformer.wte.weight") };
        ctx->CSSetShaderResources(0, 2, srvs);
        zero(logits_buf_.Get(), 0, V);
        auto uv0 = makeUAV(logits_buf_.Get(), 0, V);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((V + 15) / 16, 1, 1); clearViews(1, 2);
    }

    // 5. Loss + dlogits: logits → loss_buf_, dlogits_buf_
    const uint32_t target = (uint32_t)seq[last];
    {
        struct { uint32_t V, tgt; float inv_b; uint32_t pad; } p{V, target, inv_batch, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_loss_.Get(), nullptr, 0);
        auto sv0 = makeSRV(logits_buf_.Get(), 0, V);
        ID3D11ShaderResourceView* srvs[1] = { sv0.Get() };
        ctx->CSSetShaderResources(0, 1, srvs);
        auto uv0 = makeUAV(dlogits_buf_.Get(), 0, V);
        auto uv1 = makeUAV(loss_buf_.Get(),    0, 1);
        ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch(1, 1, 1); clearViews(2, 1);
    }

    // ══════════════════════════ BACKWARD PASS ════════════════════════════════

    // Helper: dispatch matmul_bwd_dA  (dA[M,K] += dC[M,N] @ B[K,N].T)
    // M,K,N cbuffer; SRVs: A(t0), B(t1), dC(t2); UAVs: dA(u0), dB_dummy(u1)
    auto mmBwdA = [&](uint32_t M, uint32_t K, uint32_t N,
                      ID3D11ShaderResourceView* A_srv, ID3D11ShaderResourceView* B_srv,
                      ID3D11ShaderResourceView* dC_srv,
                      ID3D11UnorderedAccessView* dA_uav) {
        struct { uint32_t M, K, N, pad; } p{M, K, N, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_bwd_dA_.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[3] = { A_srv, B_srv, dC_srv };
        ctx->CSSetShaderResources(0, 3, srvs);
        // u0=dA (written), u1=dummy (not written by CSMain_dA)
        auto dummy_uav = makeUAV(loss_buf_.Get(), 0, 1);
        ID3D11UnorderedAccessView* uvs[2] = { dA_uav, dummy_uav.Get() };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch((K + 15) / 16, (M + 15) / 16, 1); clearViews(2, 3);
    };

    // Helper: dispatch matmul_bwd_dB  (dB[K,N] += A[M,K].T @ dC[M,N])
    auto mmBwdB = [&](uint32_t M, uint32_t K, uint32_t N,
                      ID3D11ShaderResourceView* A_srv,
                      ID3D11ShaderResourceView* dC_srv,
                      ID3D11UnorderedAccessView* dB_uav) {
        struct { uint32_t M, K, N, pad; } p{M, K, N, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_bwd_dB_.Get(), nullptr, 0);
        // B is not needed by CSMain_dB but must bind something valid
        ID3D11ShaderResourceView* srvs[3] = { A_srv, nullptr, dC_srv };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto dummy_uav = makeUAV(loss_buf_.Get(), 0, 1);
        ID3D11UnorderedAccessView* uvs[2] = { dummy_uav.Get(), dB_uav };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch((N + 15) / 16, (K + 15) / 16, 1); clearViews(2, 3);
    };

    // Helper: accumulate bias gradient: dbias[N] += sum_rows(dC[M,N])
    auto biasBwd = [&](uint32_t M, uint32_t N,
                       ID3D11ShaderResourceView* dC_srv,
                       ID3D11UnorderedAccessView* dbias_uav) {
        struct { uint32_t M, N, pad[2]; } p{M, N};
        setCB(&p, 16);
        ctx->CSSetShader(cs_bias_bwd_.Get(), nullptr, 0);
        ID3D11ShaderResourceView* srvs[1] = { dC_srv };
        ctx->CSSetShaderResources(0, 1, srvs);
        ID3D11UnorderedAccessView* uvs[1] = { dbias_uav };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((N + 255) / 256, 1, 1); clearViews(1, 1);
    };

    // Helper: layernorm backward
    // dout → dh_buf_ (+=), dgamma, dbeta
    auto lnBwd = [&](uint32_t xh_off, ID3D11Buffer* xhat_buf,
                     ID3D11ShaderResourceView* gamma_srv,
                     uint32_t is_off, ID3D11Buffer* inv_std_buf,
                     ID3D11UnorderedAccessView* dgamma_uav,
                     ID3D11UnorderedAccessView* dbeta_uav) {
        struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
        setCB(&p, 16);
        ctx->CSSetShader(cs_lnorm_bwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(xhat_buf,      xh_off, S * E);
        auto sv2 = makeSRV(dh_buf_.Get(), 0,      S * E);  // dout = current dh_buf_
        auto sv3 = makeSRV(inv_std_buf,   is_off, S);
        ID3D11ShaderResourceView* srvs[4] = { sv0.Get(), gamma_srv, sv2.Get(), sv3.Get() };
        ctx->CSSetShaderResources(0, 4, srvs);
        auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);  // dx (+=)
        ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), dgamma_uav, dbeta_uav };
        ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
        ctx->Dispatch(S, 1, 1); clearViews(3, 4);
    };

    // 6. LM head backward
    // 6a. d_wte += dlogits[V,1] @ lnf_y[last,E]   (outer product, M=V, K=1, N=E)
    {
        struct { uint32_t M, K, N, use_bias; } p{V, 1, E, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(dlogits_buf_.Get(), 0,       V);  // A [V,1]
        auto sv1 = makeSRV(lnf_y_buf_.Get(),  last * E, E);  // B [1,E]
        ID3D11ShaderResourceView* srvs[3] = { sv0.Get(), sv1.Get(), nullptr };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto uv0 = makeUAV(params_[param_idx_.at("transformer.wte.weight")].g_buf.Get(), 0,
                           params_[param_idx_.at("transformer.wte.weight")].numel);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((E + 15) / 16, (V + 15) / 16, 1); clearViews(1, 3);
    }

    // 6b. d_lnf_y[last] = wte @ dlogits  (M=1, K=V, N=E, no bias)
    //     → dh_buf_[last*E .. +E]; all other positions stay 0
    zero(dh_buf_.Get(), 0, S * E);
    {
        struct { uint32_t M, K, N, use_bias; } p{1, V, E, 0};
        setCB(&p, 16);
        ctx->CSSetShader(cs_matmul_fwd_.Get(), nullptr, 0);
        auto sv0 = makeSRV(dlogits_buf_.Get(), 0, V);  // A [1,V]
        // wte w_buf SRV (not g_buf) — wSRV() returns the pre-built SRV for w_buf
        ID3D11ShaderResourceView* srvs[3] = { sv0.Get(), wSRV("transformer.wte.weight"), nullptr };
        ctx->CSSetShaderResources(0, 3, srvs);
        auto uv0 = makeUAV(dh_buf_.Get(), last * E, E);
        ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
        ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
        ctx->Dispatch((E + 15) / 16, 1, 1); clearViews(1, 3);
    }

    // 7. Final LN backward: dh_buf_ [dout only at last] → dh_buf_ [dx, +=]
    //    Also accumulates d_gamma_lnf, d_beta_lnf
    {
        auto dg_uav = makeUAV(params_[param_idx_.at("transformer.ln_f.weight")].g_buf.Get(),
                              0, E);
        auto db_uav = makeUAV(params_[param_idx_.at("transformer.ln_f.bias")].g_buf.Get(),
                              0, E);
        lnBwd(0, xhat_lnf_buf_.Get(), wSRV("transformer.ln_f.weight"),
              0, inv_std_lnf_.Get(), dg_uav.Get(), db_uav.Get());
    }

    // 8. Layer backward loop (NL-1 downto 0)
    for (int32_t li = (int32_t)NL - 1; li >= 0; --li) {
        const uint32_t l   = (uint32_t)li;
        const std::string pfx = "transformer.h." + std::to_string(l) + ".";
        const uint32_t qoff  = l * S * 3 * E;
        const uint32_t Poff  = l * H * S * S;
        const uint32_t loff1 = l * S * E;
        const uint32_t loff2 = l * S * E;
        const uint32_t aoff  = l * S * E;
        const uint32_t mpre  = l * S * F;
        const uint32_t mgel  = l * S * F;
        const uint32_t is1   = l * S;
        const uint32_t is2   = l * S;

        // === MLP BACKWARD ===
        // dh_buf_ = gradient into h[l+1]
        // Residual: d_h_mid = dh_buf_ (skip path), d_mlp_out = dh_buf_

        // 8a. d_c_proj_mlp_w += mlp_gelu[l].T @ dh_buf_   (dB: M=S, K=F, N=E)
        {
            auto A_srv  = makeSRV(mlp_gelu_buf_.Get(), mgel, S * F);
            auto dC_srv = makeSRV(dh_buf_.Get(),       0,    S * E);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "mlp.c_proj.weight")].g_buf.Get(),
                                  0, F * E);
            mmBwdB(S, F, E, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8b. d_c_proj_mlp_b += sum_rows(dh_buf_)
        {
            auto dC_srv  = makeSRV(dh_buf_.Get(), 0, S * E);
            auto db_uav  = makeUAV(params_[param_idx_.at(pfx + "mlp.c_proj.bias")].g_buf.Get(),
                                   0, E);
            biasBwd(S, E, dC_srv.Get(), db_uav.Get());
        }

        // 8c. d_mlp_gelu = dh_buf_ @ c_proj_mlp_w.T   (dA: M=S, K=F, N=E → d_mlp_buf_)
        zero(d_mlp_buf_.Get(), 0, S * F);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "mlp.c_proj.weight")].w_buf.Get(),
                                  0, F * E);
            auto dC_srv = makeSRV(dh_buf_.Get(), 0, S * E);
            auto dA_uav = makeUAV(d_mlp_buf_.Get(), 0, S * F);
            mmBwdA(S, F, E, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }

        // 8d. GELU backward: d_mlp_pre = d_mlp_buf_ * gelu'(mlp_pre[l])
        //     in-place: d_mlp_buf_ (+=)
        {
            struct { uint32_t numel, pad[3]; } p{S * F};
            setCB(&p, 16);
            ctx->CSSetShader(cs_gelu_bwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(mlp_pre_buf_.Get(), mpre, S * F);  // pre-GELU
            auto sv1 = makeSRV(d_mlp_buf_.Get(),   0,    S * F);  // upstream grad
            ID3D11ShaderResourceView* srvs[2] = { sv0.Get(), sv1.Get() };
            ctx->CSSetShaderResources(0, 2, srvs);
            auto uv0 = makeUAV(d_mlp_buf_.Get(), 0, S * F);
            ID3D11UnorderedAccessView* uvs[1] = { uv0.Get() };
            ctx->CSSetUnorderedAccessViews(0, 1, uvs, nullptr);
            ctx->Dispatch((S * F + 255) / 256, 1, 1); clearViews(1, 2);
        }

        // 8e. d_c_fc_w += ln2_y[l].T @ d_mlp_buf_   (dB: M=S, K=E, N=F)
        {
            auto A_srv  = makeSRV(ln2_y_buf_.Get(), loff2, S * E);
            auto dC_srv = makeSRV(d_mlp_buf_.Get(), 0,     S * F);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "mlp.c_fc.weight")].g_buf.Get(),
                                  0, E * F);
            mmBwdB(S, E, F, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8f. d_c_fc_b += sum_rows(d_mlp_buf_)
        {
            auto dC_srv = makeSRV(d_mlp_buf_.Get(), 0, S * F);
            auto db_uav = makeUAV(params_[param_idx_.at(pfx + "mlp.c_fc.bias")].g_buf.Get(),
                                  0, F);
            biasBwd(S, F, dC_srv.Get(), db_uav.Get());
        }

        // 8g. d_ln2_y = d_mlp_buf_ @ c_fc_w.T   (dA: M=S, K=E, N=F)
        //     → d_qkv_buf_ first S*E elements (reused as temp)
        zero(d_qkv_buf_.Get(), 0, S * E);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "mlp.c_fc.weight")].w_buf.Get(),
                                  0, E * F);
            auto dC_srv = makeSRV(d_mlp_buf_.Get(), 0, S * F);
            auto dA_uav = makeUAV(d_qkv_buf_.Get(), 0, S * E);
            mmBwdA(S, E, F, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }
        // d_qkv_buf_[0..S*E] now holds d_ln2_y

        // 8h. LN2 backward: d_ln2_y (in d_qkv_buf_) is stored as dout in…
        //     We need to swap: currently dh_buf_ is the gradient we're propagating.
        //     LN bwd reads dout from dh_buf_ (hardcoded in lnBwd helper).
        //     So: first copy d_ln2_y to dh_buf_, then call lnBwd (which adds dx to dh_buf_).
        //     But lnBwd reads dout = current dh_buf_ (which is d_h_out/d_h_mid skip gradient).
        //     We need dout = d_ln2_y, not d_h_mid.
        //
        //     Fix: use a separate LN bwd call where dout = d_qkv_buf_[:S*E].
        //     The lnBwd helper is hardcoded to read dh_buf_ as dout — we need a different binding.
        //
        //     So: do it inline here rather than via lnBwd helper.
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            ctx->CSSetShader(cs_lnorm_bwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(xhat_ln2_buf_.Get(), loff2, S * E);  // xhat
            auto sv1 = wSRV(pfx + "ln_2.weight");                    // gamma
            auto sv2 = makeSRV(d_qkv_buf_.Get(),    0,     S * E);  // dout = d_ln2_y
            auto sv3 = makeSRV(inv_std_ln2_.Get(),  is2,   S);       // inv_std
            ID3D11ShaderResourceView* srvs[4] = { sv0.Get(), sv1, sv2.Get(), sv3.Get() };
            ctx->CSSetShaderResources(0, 4, srvs);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);  // dx += (adds to d_h_mid)
            auto uv1 = makeUAV(params_[param_idx_.at(pfx + "ln_2.weight")].g_buf.Get(), 0, E);
            auto uv2 = makeUAV(params_[param_idx_.at(pfx + "ln_2.bias")  ].g_buf.Get(), 0, E);
            ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 4);
        }
        // dh_buf_ = d_h_mid  (gradient into h after first residual)

        // === ATTENTION BACKWARD ===

        // 8i. d_c_proj_attn_w += attn_out[l].T @ dh_buf_   (dB: M=S, K=E, N=E)
        {
            auto A_srv  = makeSRV(attn_out_buf_.Get(), aoff, S * E);
            auto dC_srv = makeSRV(dh_buf_.Get(),       0,    S * E);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_proj.weight")].g_buf.Get(),
                                  0, E * E);
            mmBwdB(S, E, E, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8j. d_c_proj_attn_b += sum_rows(dh_buf_)
        {
            auto dC_srv = makeSRV(dh_buf_.Get(), 0, S * E);
            auto db_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_proj.bias")].g_buf.Get(),
                                  0, E);
            biasBwd(S, E, dC_srv.Get(), db_uav.Get());
        }

        // 8k. d_attn_out = dh_buf_ @ c_proj_attn_w.T   (dA: M=S, K=E, N=E)
        //     → lnf_y_buf_ (reused as temp, forward done)
        zero(lnf_y_buf_.Get(), 0, S * E);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "attn.c_proj.weight")].w_buf.Get(),
                                  0, E * E);
            auto dC_srv = makeSRV(dh_buf_.Get(), 0, S * E);
            auto dA_uav = makeUAV(lnf_y_buf_.Get(), 0, S * E);
            mmBwdA(S, E, E, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }
        // lnf_y_buf_ = d_attn_out [S, E]

        // 8l. Attention backward (3-pass per head, writes to d_qkv_buf_ [S,3E])
        zero(d_qkv_buf_.Get(), 0, S * 3 * E);
        for (uint32_t h = 0; h < H; ++h) {
            // Zero per-head temps
            zero(dP_tmp_buf_.Get(), 0, S * S);
            zero(dot_row_buf_.Get(), 0, S);

            struct { uint32_t S, D, E, h; float scale; uint32_t pad[3]; }
                abp{S, D, E, h, scale};
            setCB(&abp, 32);

            auto qkv_srv  = makeSRV(qkv_buf_.Get(),    qoff,             S * 3 * E);
            auto P_srv    = makeSRV(P_buf_.Get(),       Poff + h * S * S, S * S);
            auto dout_srv = makeSRV(lnf_y_buf_.Get(),  0,                S * E);
            auto dqkv_uav = makeUAV(d_qkv_buf_.Get(),  0,                S * 3 * E);
            auto dP_uav   = makeUAV(dP_tmp_buf_.Get(), 0,                S * S);
            auto dr_uav   = makeUAV(dot_row_buf_.Get(), 0,               S);

            // Pass 1: dV and dP_tmp
            ctx->CSSetShader(cs_attn_bwd_dvdp_.Get(), nullptr, 0);
            {
                ID3D11ShaderResourceView* srvs[3] = { qkv_srv.Get(), P_srv.Get(), dout_srv.Get() };
                ctx->CSSetShaderResources(0, 3, srvs);
                ID3D11UnorderedAccessView* uvs[3] = { dqkv_uav.Get(), dP_uav.Get(), dr_uav.Get() };
                ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            }
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);

            // Pass 2: dQ and dot_row
            setCB(&abp, 32);
            ctx->CSSetShader(cs_attn_bwd_dq_.Get(), nullptr, 0);
            {
                ID3D11ShaderResourceView* srvs[3] = { qkv_srv.Get(), P_srv.Get(), dout_srv.Get() };
                ctx->CSSetShaderResources(0, 3, srvs);
                ID3D11UnorderedAccessView* uvs[3] = { dqkv_uav.Get(), dP_uav.Get(), dr_uav.Get() };
                ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            }
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);

            // Pass 3: dK
            setCB(&abp, 32);
            ctx->CSSetShader(cs_attn_bwd_dk_.Get(), nullptr, 0);
            {
                ID3D11ShaderResourceView* srvs[3] = { qkv_srv.Get(), P_srv.Get(), dout_srv.Get() };
                ctx->CSSetShaderResources(0, 3, srvs);
                ID3D11UnorderedAccessView* uvs[3] = { dqkv_uav.Get(), dP_uav.Get(), dr_uav.Get() };
                ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            }
            ctx->Dispatch(S, 1, 1); clearViews(3, 3);
        }
        // d_qkv_buf_ [S, 3E] now holds dQ, dK, dV (interleaved)

        // 8m. d_c_attn_w += ln1_y[l].T @ d_qkv_buf_   (dB: M=S, K=E, N=3E)
        {
            auto A_srv  = makeSRV(ln1_y_buf_.Get(),  loff1, S * E);
            auto dC_srv = makeSRV(d_qkv_buf_.Get(),  0,     S * 3 * E);
            auto dB_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_attn.weight")].g_buf.Get(),
                                  0, E * 3 * E);
            mmBwdB(S, E, 3 * E, A_srv.Get(), dC_srv.Get(), dB_uav.Get());
        }

        // 8n. d_c_attn_b += sum_rows(d_qkv_buf_)
        {
            auto dC_srv = makeSRV(d_qkv_buf_.Get(), 0, S * 3 * E);
            auto db_uav = makeUAV(params_[param_idx_.at(pfx + "attn.c_attn.bias")].g_buf.Get(),
                                  0, 3 * E);
            biasBwd(S, 3 * E, dC_srv.Get(), db_uav.Get());
        }

        // 8o. d_ln1_y = d_qkv_buf_ @ c_attn_w.T   (dA: M=S, K=E, N=3E)
        //     → lnf_y_buf_ (reused again as temp)
        zero(lnf_y_buf_.Get(), 0, S * E);
        {
            auto B_srv  = makeSRV(params_[param_idx_.at(pfx + "attn.c_attn.weight")].w_buf.Get(),
                                  0, E * 3 * E);
            auto dC_srv = makeSRV(d_qkv_buf_.Get(), 0, S * 3 * E);
            auto dA_uav = makeUAV(lnf_y_buf_.Get(), 0, S * E);
            mmBwdA(S, E, 3 * E, nullptr, B_srv.Get(), dC_srv.Get(), dA_uav.Get());
        }
        // lnf_y_buf_ = d_ln1_y [S, E]

        // 8p. LN1 backward: dout = d_ln1_y (in lnf_y_buf_), dx += to dh_buf_
        {
            struct { uint32_t E, S; float eps; uint32_t pad; } p{E, S, eps};
            setCB(&p, 16);
            ctx->CSSetShader(cs_lnorm_bwd_.Get(), nullptr, 0);
            auto sv0 = makeSRV(xhat_ln1_buf_.Get(), loff1, S * E);  // xhat
            auto sv1 = wSRV(pfx + "ln_1.weight");                    // gamma
            auto sv2 = makeSRV(lnf_y_buf_.Get(),    0,     S * E);  // dout = d_ln1_y
            auto sv3 = makeSRV(inv_std_ln1_.Get(),  is1,   S);       // inv_std
            ID3D11ShaderResourceView* srvs[4] = { sv0.Get(), sv1, sv2.Get(), sv3.Get() };
            ctx->CSSetShaderResources(0, 4, srvs);
            auto uv0 = makeUAV(dh_buf_.Get(), 0, S * E);  // dx += (d_h[l])
            auto uv1 = makeUAV(params_[param_idx_.at(pfx + "ln_1.weight")].g_buf.Get(), 0, E);
            auto uv2 = makeUAV(params_[param_idx_.at(pfx + "ln_1.bias")  ].g_buf.Get(), 0, E);
            ID3D11UnorderedAccessView* uvs[3] = { uv0.Get(), uv1.Get(), uv2.Get() };
            ctx->CSSetUnorderedAccessViews(0, 3, uvs, nullptr);
            ctx->Dispatch(S, 1, 1); clearViews(3, 4);
        }
        // dh_buf_ = d_h[l]  (gradient into layer l input)
    } // end layer backward loop

    // 9. Embedding backward: dh_buf_ → d_wte (+=), d_wpe (+=)
    {
        struct { uint32_t S, E, pad[2]; } p{S, E};
        setCB(&p, 16);
        ctx->CSSetShader(cs_embed_bwd_.Get(), nullptr, 0);
        auto tsrv = makeSRVi(tokens_buf_.Get(), 0, S);
        auto dh_srv = makeSRV(dh_buf_.Get(), 0, S * E);
        ID3D11ShaderResourceView* srvs[2] = { tsrv.Get(), dh_srv.Get() };
        ctx->CSSetShaderResources(0, 2, srvs);
        auto uv0 = makeUAV(params_[param_idx_.at("transformer.wte.weight")].g_buf.Get(),
                            0, params_[param_idx_.at("transformer.wte.weight")].numel);
        auto uv1 = makeUAV(params_[param_idx_.at("transformer.wpe.weight")].g_buf.Get(),
                            0, params_[param_idx_.at("transformer.wpe.weight")].numel);
        ID3D11UnorderedAccessView* uvs[2] = { uv0.Get(), uv1.Get() };
        ctx->CSSetUnorderedAccessViews(0, 2, uvs, nullptr);
        ctx->Dispatch(E, 1, 1); clearViews(2, 2);
    }

    return readbackLoss();
}
