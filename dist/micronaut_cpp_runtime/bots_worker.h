#pragma once
#include "common.h"

namespace micronaut {

// Persistent bots.py subprocess with bidirectional JSON-line protocol.
// One instance per micronaut skin; kept alive for the lifetime of the runtime.
// Thread-safe: call() can be invoked from multiple threads sequentially
// (internal mutex serializes access to the subprocess pipes).
class BotsWorker {
public:
    BotsWorker() = default;
    ~BotsWorker();

    // Spawn "python bots.py" in the given working directory.
    // Returns true on success.
    bool start(const std::string& work_dir);

    // Send a JSON-line request to bots.py; return the JSON-line response.
    // task: the bots.py task name (e.g. "health", "question", "expand")
    // payload_json: JSON object string for the payload field
    std::string call(const std::string& task, const std::string& payload_json = "{}");

    bool running() const { return running_; }
    const std::string& last_error() const { return last_error_; }

private:
    std::string read_line(int timeout_ms = 5000);
    void write_line(const std::string& line);

#ifdef _WIN32
    void* proc_handle_  = nullptr; // HANDLE
    void* thread_handle_= nullptr; // HANDLE
    void* stdin_write_  = nullptr; // HANDLE
    void* stdout_read_  = nullptr; // HANDLE
#else
    int stdin_fd_  = -1;
    int stdout_fd_ = -1;
    int pid_       = -1;
#endif

    bool running_ = false;
    std::mutex mu_;
    std::string last_error_;
};

// Global bots worker singleton for this skin instance.
BotsWorker& bots_worker();

} // namespace micronaut
