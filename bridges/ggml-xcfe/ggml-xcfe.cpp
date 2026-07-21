// ggml-xcfe.cpp — KHΛNARY XCFE backend for ggml (structure mirrors ggml-blas).
//
// It is a genuine, distinct ggml backend: its own name ("XCFE") and GUID, registered through
// ggml_backend_xcfe_reg(). Like the BLAS backend it reuses the CPU host-memory buffer type and
// claims only GGML_OP_MUL_MAT, so ggml's scheduler assigns matmuls here and the rest to CPU.
//
// STATUS (honest): graph_compute currently runs MUL_MAT with a portable F32 reference GEMM —
// this is the PLACEHOLDER for the KHΛNARY glyph dispatch. The single function
// `ggml_backend_xcfe_gemm_f32` below is the seam where the verified D3D11 cs_5_0 `G_MATMUL`
// kernel (or DirectML's DML_OPERATOR_GEMM, ~4.9x faster on the HD 4600) plugs in. Until that
// lands, results are correct but computed on the CPU.
#include "ggml-impl.h"
#include "ggml-xcfe.h"
#include "ggml-backend-impl.h"

#include <cstring>
#include <cstdlib>

struct ggml_backend_xcfe_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
};

// ---------------------------------------------------------------------------------------------
// The KHΛNARY compute seam. dst = src1 @ src0^T in ggml's MUL_MAT convention:
//   src0: [ne00=K, ne01=N, ne02, ne03]   src1: [ne10=K, ne11=M, ne12, ne13]
//   dst : [ne0=N, ne1=M, ne2, ne3]  with dst[i3,i2,m,n] = sum_k src0[i03,i02,n,k]*src1[i13,i12,m,k]
// supports_op guarantees F32 + contiguous inputs, so the K dimension is unit-stride here.
// >>> REPLACE THIS BODY with the KHΛNARY glyph GEMM (D3D11 cs_5_0 G_MATMUL / DirectML). <<<
// ---------------------------------------------------------------------------------------------
static void ggml_backend_xcfe_gemm_f32(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t K = ne00;      // shared inner dim
    const int64_t N = ne01;      // rows of src0
    const int64_t M = ne11;      // rows of src1

    // src0 broadcasts over the batch dims of src1
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t i03 = i3 / r3;
            const int64_t i02 = i2 / r2;
            const char * a_base = (const char *) src0->data + i03*nb03 + i02*nb02;
            const char * b_base = (const char *) src1->data + i3 *nb13 + i2 *nb12;
            char       * d_base = (      char *) dst->data  + i3 *nb3  + i2 *nb2;
            for (int64_t m = 0; m < M; m++) {
                const float * b = (const float *)(b_base + m*nb11);
                float       * d = (      float *)(d_base + m*nb1);
                for (int64_t n = 0; n < N; n++) {
                    const float * a = (const float *)(a_base + n*nb01);
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; k++) acc += a[k] * b[k];
                    *(float *)((char *)d + n*nb0) = acc;
                }
            }
        }
    }
}

// backend interface

static const char * ggml_backend_xcfe_get_name(ggml_backend_t backend) {
    return "XCFE";
    GGML_UNUSED(backend);
}

static void ggml_backend_xcfe_free(ggml_backend_t backend) {
    ggml_backend_xcfe_context * ctx = (ggml_backend_xcfe_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xcfe_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_xcfe_gemm_f32(node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static struct ggml_backend_i xcfe_backend_i = {
    /* .get_name                = */ ggml_backend_xcfe_get_name,
    /* .free                    = */ ggml_backend_xcfe_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xcfe_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xcfe_guid(void) {
    // distinct from every upstream backend GUID
    static ggml_guid guid = { 0x78, 0x63, 0x66, 0x65, 0x4b, 0x48, 0x4c, 0x21, 0x67, 0x67, 0x6d, 0x6c, 0x78, 0x63, 0x66, 0x65 };
    return &guid;
}

ggml_backend_t ggml_backend_xcfe_init(void) {
    ggml_backend_xcfe_context * ctx = new ggml_backend_xcfe_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xcfe_guid(),
        /* .iface   = */ xcfe_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xcfe_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xcfe(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xcfe_guid());
}

void ggml_backend_xcfe_set_n_threads(ggml_backend_t backend_xcfe, int n_threads) {
    GGML_ASSERT(ggml_backend_is_xcfe(backend_xcfe));
    ggml_backend_xcfe_context * ctx = (ggml_backend_xcfe_context *)backend_xcfe->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_xcfe_device_get_name(ggml_backend_dev_t dev) {
    return "XCFE";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_xcfe_device_get_description(ggml_backend_dev_t dev) {
    return "KHANARY glyph runtime (MUL_MAT via G_MATMUL; CPU reference until GPU dispatch is wired)";
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xcfe_device_get_type(ggml_backend_dev_t dev) {
    // ACCEL (not IGPU): honest label while graph_compute is the CPU reference. Flip to
    // GGML_BACKEND_DEVICE_TYPE_IGPU once the D3D11 cs_5_0 / DirectML dispatch is wired.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xcfe_device_get_name(dev);
    props->description = ggml_backend_xcfe_device_get_description(dev);
    props->type        = ggml_backend_xcfe_device_get_type(dev);
    ggml_backend_xcfe_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = NULL;
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xcfe_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xcfe_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xcfe_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xcfe_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xcfe_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const int64_t ne10 = src1->ne[0];
            const int64_t ne0  = op->ne[0];
            const int64_t ne1  = op->ne[1];
            const int64_t min_batch = 32;   // like BLAS: only worth it for large matmuls

            // F32-only for the reference GEMM (the glyph kernel is F32 too). Quantized src0
            // would need a dequant step — deferred (see the GGUF dequant->.stb note).
            return ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src0->type == GGML_TYPE_F32 &&
                   src1->type == GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch);
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xcfe_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xcfe_device_i = {
    /* .get_name             = */ ggml_backend_xcfe_device_get_name,
    /* .get_description      = */ ggml_backend_xcfe_device_get_description,
    /* .get_memory           = */ ggml_backend_xcfe_device_get_memory,
    /* .get_type             = */ ggml_backend_xcfe_device_get_type,
    /* .get_props            = */ ggml_backend_xcfe_device_get_props,
    /* .init_backend         = */ ggml_backend_xcfe_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xcfe_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xcfe_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xcfe_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xcfe_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xcfe_reg_get_name(ggml_backend_reg_t reg) {
    return "XCFE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_xcfe_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xcfe_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xcfe_device = {
        /* .iface   = */ ggml_backend_xcfe_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xcfe_device;
    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_xcfe_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_xcfe_set_n_threads;
    }
    return NULL;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_xcfe_reg_i = {
    /* .get_name         = */ ggml_backend_xcfe_reg_get_name,
    /* .get_device_count = */ ggml_backend_xcfe_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xcfe_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xcfe_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xcfe_reg(void) {
    static struct ggml_backend_reg ggml_backend_xcfe_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xcfe_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xcfe_reg;
}
