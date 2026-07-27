// ggml-xcfe.cpp — KHANARY-native ggml backend (the KHΛNARY slot in the ggml registry).
//
// A real, distinct, registering backend that BUILDS with only the CPU backend present. It mirrors
// the minimal-real shape of ggml-blas: buffers delegate to the CPU backend, and `supports_op`
// currently returns false so the scheduler never routes ops here (everything falls back to CPU).
// `graph_compute` is therefore a no-op success with ZERO external dependencies. The K'UHUL
// glyph-lowering compute (MUL_MAT -> KHANARY glyph kernels) replaces the stub later.
#include "ggml-impl.h"
#include "ggml-xcfe.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"   // ggml_backend_cpu_buffer_type / ggml_backend_buft_is_host

// ---------------------------------------------------------------------------- backend interface

static const char * ggml_backend_xcfe_get_name(ggml_backend_t backend) {
    return "XCFE";
    GGML_UNUSED(backend);
}

static void ggml_backend_xcfe_free(ggml_backend_t backend) {
    delete backend;
}

static enum ggml_status ggml_backend_xcfe_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Stub milestone: supports_op returns false, so the scheduler never routes ops to this backend.
    // The K'UHUL glyph-lowering compute lands here next (MUL_MAT -> KHANARY glyph kernels).
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
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
    // distinct KHANARY guid (spells nothing; just unique vs other backends)
    static ggml_guid guid = { 0x4b, 0x48, 0x41, 0x4e, 0x41, 0x52, 0x59, 0x00, 0x78, 0x63, 0x66, 0x65, 0x11, 0x22, 0x33, 0x44 };
    return &guid;
}

ggml_backend_t ggml_backend_xcfe_init(void) {
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xcfe_guid(),
        /* .iface   = */ xcfe_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xcfe_reg(), 0),
        /* .context = */ nullptr,
    };
    return backend;
}

bool ggml_backend_is_xcfe(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xcfe_guid());
}

// ---------------------------------------------------------------------------- device interface

static const char * ggml_backend_xcfe_device_get_name(ggml_backend_dev_t dev) {
    return "XCFE";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_xcfe_device_get_description(ggml_backend_dev_t dev) {
    return "KHANARY XCFE (K'UHUL glyph backend, stub: ops fall back to CPU)";
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xcfe_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_xcfe_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xcfe_device_get_name(dev);
    props->description = ggml_backend_xcfe_device_get_description(dev);
    props->type        = ggml_backend_xcfe_device_get_type(dev);
    ggml_backend_xcfe_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xcfe_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xcfe_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xcfe_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();   // delegate storage to the CPU backend
    GGML_UNUSED(dev);
}

static bool ggml_backend_xcfe_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Stub milestone: claim nothing -> the scheduler keeps every op on CPU. The glyph-lowering
    // vtable will return true here for the ops KHANARY kernels handle (e.g. GGML_OP_MUL_MAT).
    return false;
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
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
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_xcfe_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xcfe_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------- reg interface

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

static const struct ggml_backend_reg_i ggml_backend_xcfe_reg_i = {
    /* .get_name         = */ ggml_backend_xcfe_reg_get_name,
    /* .get_device_count = */ ggml_backend_xcfe_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xcfe_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_xcfe_reg(void) {
    static struct ggml_backend_reg ggml_backend_xcfe_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xcfe_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xcfe_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xcfe_reg)
