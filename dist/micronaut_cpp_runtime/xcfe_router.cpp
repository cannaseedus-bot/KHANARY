#include "xcfe_router.h"
#include "chat.h"
#include "process.h"
#include "batches.h"
#include "data_index.h"
#include "kuhule.h"
#include "phase_router.h"

namespace micronaut {

void XcfeRouter::register_handler(const std::string& op_prefix, Handler handler) {
    handlers_[op_prefix] = handler;
}

std::string XcfeRouter::route_name(const std::string& op) const {
    for (const auto& [prefix, handler] : handlers_) {
        if (op.rfind(prefix, 0) == 0) return prefix;
    }
    return "";
}

MicronautResult XcfeRouter::route(const MicronautOp& op) const {
    auto start = Clock::now();
    for (const auto& [prefix, handler] : handlers_) {
        if (op.op.rfind(prefix, 0) == 0) {
            return handler(op);
        }
    }

    MicronautResult r;
    r.id = op.id;
    r.op = op.op;
    r.ok = false;
    r.error_code = "XCFE_ROUTE_NOT_FOUND";
    r.error_message = "No route registered for op: " + op.op;
    auto end = Clock::now();
    r.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return r;
}

XcfeRouter& xcfe_router() {
    static XcfeRouter router;
    return router;
}

void register_builtin_handlers() {
    static bool done = false;
    if (done) return;
    done = true;

    xcfe_router().register_handler("chat.", chat_execute);
    xcfe_router().register_handler("batch.", batch_execute);
    xcfe_router().register_handler("data.", data_execute);
    xcfe_router().register_handler("kuhule.", kuhule_execute);
    xcfe_router().register_handler("dxsk.", process_execute);
    xcfe_router().register_handler("scx.", process_execute);
    xcfe_router().register_handler("node.", process_execute);
    xcfe_router().register_handler("dotnet.", process_execute);
    xcfe_router().register_handler("java.", process_execute);
    xcfe_router().register_handler("gpt2.", process_execute);

    // Phase/fold/glyph/micronaut ops — routed through PhaseRouter
    auto phase_dispatch = [](const MicronautOp& op) { return phase_router().route(op); };
    xcfe_router().register_handler("fold.",          phase_dispatch);
    xcfe_router().register_handler("phase.",         phase_dispatch);
    xcfe_router().register_handler("glyph.",         phase_dispatch);
    xcfe_router().register_handler("reinforcement.", phase_dispatch);
    xcfe_router().register_handler("micronaut.",     phase_dispatch);
}

} // namespace micronaut
