#include "bots_worker.h"
#include "output.h"
#include <sstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace micronaut {

// ---------------------------------------------------------------------------
// Win32 implementation
// ---------------------------------------------------------------------------

#ifdef _WIN32

bool BotsWorker::start(const std::string& work_dir) {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return true;

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE stdin_r, stdin_w, stdout_r, stdout_w;

    if (!CreatePipe(&stdin_r, &stdin_w, &sa, 0)) {
        last_error_ = "CreatePipe(stdin) failed";
        return false;
    }
    if (!CreatePipe(&stdout_r, &stdout_w, &sa, 0)) {
        CloseHandle(stdin_r); CloseHandle(stdin_w);
        last_error_ = "CreatePipe(stdout) failed";
        return false;
    }

    // Don't inherit the ends the parent keeps
    SetHandleInformation(stdin_w,  HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdout_r, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = stdin_r;
    si.hStdOutput = stdout_w;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi{};
    std::string cmd = "python bots.py";
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr, cmd_buf.data(),
        nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW,
        nullptr, work_dir.empty() ? nullptr : work_dir.c_str(),
        &si, &pi
    );

    CloseHandle(stdin_r);
    CloseHandle(stdout_w);

    if (!ok) {
        CloseHandle(stdin_w); CloseHandle(stdout_r);
        last_error_ = "CreateProcess(python bots.py) failed, error=" + std::to_string(GetLastError());
        return false;
    }

    CloseHandle(pi.hThread);
    proc_handle_   = pi.hProcess;
    thread_handle_ = nullptr;
    stdin_write_   = stdin_w;
    stdout_read_   = stdout_r;
    running_ = true;

    output().log("info", "[bots_worker] started python bots.py in " + (work_dir.empty() ? "." : work_dir));
    return true;
}

void BotsWorker::write_line(const std::string& line) {
    std::string l = line + "\n";
    DWORD written;
    WriteFile(static_cast<HANDLE>(stdin_write_), l.c_str(), static_cast<DWORD>(l.size()), &written, nullptr);
}

std::string BotsWorker::read_line(int timeout_ms) {
    std::string result;
    HANDLE h = static_cast<HANDLE>(stdout_read_);
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        DWORD avail = 0;
        if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr)) break;

        if (avail == 0) {
            if (Clock::now() > deadline) { last_error_ = "bots.py read timeout"; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        char buf[4096];
        DWORD read = 0;
        DWORD to_read = std::min<DWORD>(avail, sizeof(buf) - 1);
        if (!ReadFile(h, buf, to_read, &read, nullptr) || read == 0) break;
        buf[read] = '\0';
        result += buf;
        if (result.find('\n') != std::string::npos) break;
    }

    // Trim trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

BotsWorker::~BotsWorker() {
    if (!running_) return;
    if (stdin_write_) { CloseHandle(static_cast<HANDLE>(stdin_write_)); stdin_write_ = nullptr; }
    if (stdout_read_) { CloseHandle(static_cast<HANDLE>(stdout_read_)); stdout_read_ = nullptr; }
    if (proc_handle_) {
        WaitForSingleObject(static_cast<HANDLE>(proc_handle_), 2000);
        CloseHandle(static_cast<HANDLE>(proc_handle_)); proc_handle_ = nullptr;
    }
    running_ = false;
}

// ---------------------------------------------------------------------------
// POSIX implementation (non-Windows)
// ---------------------------------------------------------------------------

#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>

bool BotsWorker::start(const std::string& work_dir) {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return true;

    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) || pipe(out_pipe)) {
        last_error_ = "pipe() failed";
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        last_error_ = "fork() failed";
        return false;
    }
    if (pid == 0) {
        // child
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        if (!work_dir.empty()) chdir(work_dir.c_str());
        execlp("python3", "python3", "bots.py", nullptr);
        execlp("python",  "python",  "bots.py", nullptr);
        _exit(1);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    stdin_fd_  = in_pipe[1];
    stdout_fd_ = out_pipe[0];
    pid_       = pid;
    running_   = true;
    return true;
}

void BotsWorker::write_line(const std::string& line) {
    std::string l = line + "\n";
    ::write(stdin_fd_, l.c_str(), l.size());
}

std::string BotsWorker::read_line(int timeout_ms) {
    std::string result;
    auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    char c;
    while (Clock::now() < deadline) {
        fd_set fds; FD_ZERO(&fds); FD_SET(stdout_fd_, &fds);
        timeval tv{0, 5000};
        if (select(stdout_fd_+1, &fds, nullptr, nullptr, &tv) > 0) {
            if (::read(stdout_fd_, &c, 1) == 1) {
                if (c == '\n') break;
                result += c;
            }
        }
    }
    return result;
}

BotsWorker::~BotsWorker() {
    if (!running_) return;
    close(stdin_fd_); close(stdout_fd_);
    waitpid(pid_, nullptr, 0);
    running_ = false;
}
#endif

// ---------------------------------------------------------------------------
// Shared: call()
// ---------------------------------------------------------------------------

std::string BotsWorker::call(const std::string& task, const std::string& payload_json) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!running_) {
        return "{\"error\":\"bots_worker not running\",\"task\":\"" + task + "\"}";
    }

    std::string req = "{\"task\":\"" + task + "\",\"payload\":" + payload_json + "}";
    write_line(req);
    std::string resp = read_line();

    if (resp.empty()) {
        return "{\"error\":\"bots_worker timeout or empty response\",\"task\":\"" + task + "\"}";
    }
    return resp;
}

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

BotsWorker& bots_worker() {
    static BotsWorker worker;
    return worker;
}

} // namespace micronaut
