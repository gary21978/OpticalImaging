#include "matrix_storage.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <vector>

namespace rcwa_memory
{
namespace
{
enum class BackingKind { kFile, kRam };
enum class BackingPolicy { kFile, kRamOnly, kHybrid };
struct Allocation
{
    void* ptr;
    size_t bytes;
    int fd;
    BackingKind kind;
};
std::map<uintptr_t, Allocation> backing;
std::map<void*, size_t> resident;
size_t resident_bytes = 0;
std::mutex file_mutex;
thread_local bool enabled = false;
thread_local BackingPolicy policy = BackingPolicy::kFile;
Stats stats;
size_t EnvSize(const char* name, size_t fallback)
{
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    char* end = nullptr;
    errno = 0;
    const unsigned long long n = std::strtoull(s, &end, 10);
    return errno || *s == '-' || *end || !n || n > std::numeric_limits<size_t>::max()
               ? fallback : static_cast<size_t>(n);
}
size_t RamBudgetBytes()
{
    const char* s = std::getenv("RCWA_RAM_MB");
    return s && !std::strcmp(s, "0") ? 0 : EnvSize("RCWA_RAM_MB", 2048) * 1024 * 1024;
}
size_t AvailableRamBytes()
{
    // MemAvailable includes reclaimable pages, unlike the free-page count.
    FILE* info = std::fopen("/proc/meminfo", "r");
    if (info)
    {
        char line[256];
        unsigned long long kib = 0;
        while (std::fgets(line, sizeof(line), info))
            if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1)
            {
                std::fclose(info);
                return static_cast<size_t>(kib) * 1024;
            }
        std::fclose(info);
    }
    const long pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    return pages > 0 && page_size > 0 ? static_cast<size_t>(pages) * page_size : 0;
}
bool Find(const void* p, Allocation* out, size_t* offset)
{
    std::lock_guard<std::mutex> guard(file_mutex);
    auto it = backing.upper_bound(reinterpret_cast<uintptr_t>(p));
    if (it == backing.begin()) return false;
    --it;
    const auto off = reinterpret_cast<uintptr_t>(p) - it->first;
    if (off >= it->second.bytes) return false;
    *out = it->second;
    *offset = off;
    return true;
}
Status Error(const char* what)
{
    std::fprintf(stderr, "RCWA storage: %s: %s\n", what, std::strerror(errno));
    return Status::kCudaError;
}
Status IO(int fd, void* ptr, size_t bytes, size_t off, bool write)
{
    if (policy == BackingPolicy::kRamOnly)
    {
        std::fprintf(stderr, "RCWA RAM-only: access to an existing file matrix is disabled\n");
        return Status::kInvalidValue;
    }
    char* p = static_cast<char*>(ptr);
    while (bytes)
    {
        const ssize_t n = write ? pwrite(fd, p, bytes, static_cast<off_t>(off))
                                : pread(fd, p, bytes, static_cast<off_t>(off));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0)
        {
            if (!n) errno = EIO;
            return Error(write ? "pwrite" : "pread");
        }
        p += n;
        off += n;
        bytes -= n;
    }
    return Status::kSuccess;
}
Status Flush(const void* p)
{
    // These unlinked files are disposable working storage. pwrite is already
    // coherent with subsequent pread calls; durability is not required. Forcing
    // every tile to SSD and dropping the whole file cache defeats block reuse.
    const char* durable = std::getenv("RCWA_SYNC_FILES");
    if (!durable || std::strcmp(durable, "1")) return Status::kSuccess;
    Allocation a{};
    size_t off = 0;
    if (!Find(p, &a, &off) || a.kind == BackingKind::kRam) return Status::kSuccess;
    if (policy == BackingPolicy::kRamOnly) return Status::kInvalidValue;
    if (fdatasync(a.fd)) return Error("fdatasync");
    return Status::kSuccess;
}
struct Device
{
    Complex* p = nullptr;
    ~Device()
    {
        if (p) cudaFree(p);
    }
    Status Alloc(size_t n)
    {
        return cudaMalloc(&p, n * sizeof(Complex)) == cudaSuccess ? Status::kSuccess
                                                                  : Status::kCudaError;
    }
    void Reset()
    {
        if (p) cudaFree(p);
        p = nullptr;
    }
};
struct CopyStream
{
    cudaStream_t stream = nullptr;
    ~CopyStream() { if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
};
struct Fence
{
    cudaStream_t stream;
    ~Fence() { cudaStreamSynchronize(stream); }
};
struct CopyEvent
{
    cudaEvent_t event = nullptr;
    ~CopyEvent() { if (event) cudaEventDestroy(event); }
    Status Create()
    {
        return cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess
                   ? Status::kSuccess : Status::kCudaError;
    }
};
struct PinnedPool
{
    struct Entry { Complex* ptr; size_t count; int device; };
    std::vector<Entry> entries;
    ~PinnedPool() { for (const auto& entry : entries) cudaFreeHost(entry.ptr); }
};
thread_local PinnedPool staging_pool;
bool UseStagingPool()
{
    const char* value = std::getenv("RCWA_STAGING_POOL");
    return value && std::strcmp(value, "1") == 0;
}
bool UseDirectPinned()
{
    const char* value = std::getenv("RCWA_DIRECT_PINNED");
    return !value || std::strcmp(value, "0") != 0;
}
struct HostBuffer
{
    Complex* p = nullptr;
    size_t count = 0;
    bool pinned = false;
    int device = -1;
    ~HostBuffer()
    {
        Reset();
    }
    void Reset()
    {
        if (pinned && p)
        {
            if (UseStagingPool() && device >= 0)
            {
                auto smallest = std::min_element(staging_pool.entries.begin(), staging_pool.entries.end(),
                    [](const auto& a, const auto& b) { return a.count < b.count; });
                if (staging_pool.entries.size() < 4)
                    staging_pool.entries.push_back({p, count, device});
                else if (smallest != staging_pool.entries.end() && smallest->count < count)
                {
                    cudaFreeHost(smallest->ptr);
                    *smallest = {p, count, device};
                }
                else cudaFreeHost(p);
            }
            else cudaFreeHost(p);
        }
        else std::free(p);
        p = nullptr;
        count = 0;
        pinned = false;
        device = -1;
    }
    Complex* data() { return p; }
    Complex& operator[](size_t i) { return p[i]; }
    Status Alloc(size_t wanted)
    {
        if (!wanted) return Status::kSuccess;
        const size_t limit = EnvSize("RCWA_HOST_BUFFER_MB", 128) * 1024 * 1024 /
                             sizeof(Complex);
        count = std::min(wanted, limit);
        if (UseStagingPool())
        {
            if (cudaGetDevice(&device) != cudaSuccess) { cudaGetLastError(); device = -1; }
            if (device >= 0)
            {
                auto best = staging_pool.entries.end();
                for (auto it = staging_pool.entries.begin(); it != staging_pool.entries.end(); ++it)
                    if (it->device == device && it->count >= count && it->count <= limit &&
                        (best == staging_pool.entries.end() || it->count < best->count)) best = it;
                if (best != staging_pool.entries.end())
                {
                    p = best->ptr;
                    count = best->count;
                    pinned = true;
                    staging_pool.entries.erase(best);
                    return Status::kSuccess;
                }
            }
        }
        if (cudaMallocHost(&p, count * sizeof(Complex)) == cudaSuccess)
            pinned = true;
        else
        {
            cudaGetLastError();
            if (policy == BackingPolicy::kRamOnly)
            {
                p = nullptr;
                std::fprintf(stderr, "RCWA RAM-only: unable to lock staging RAM; no pageable fallback\n");
                return Status::kCudaError;
            }
            p = static_cast<Complex*>(std::malloc(count * sizeof(Complex)));
        }
        return p ? Status::kSuccess : Status::kCudaError;
    }
};
Status Blas(cublasStatus_t s)
{
    return s == CUBLAS_STATUS_SUCCESS ? Status::kSuccess : Status::kCublasError;
}
Status Cuda(cudaError_t s)
{
    if (s != cudaSuccess) std::fprintf(stderr, "RCWA storage CUDA: %s\n", cudaGetErrorString(s));
    return s == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}
Status Sync(cudaStream_t s) { return Cuda(cudaStreamSynchronize(s)); }
bool RamRect(const Complex* ptr, int64_t ld, int64_t row, int64_t col,
             int64_t rows, int64_t cols)
{
    Allocation allocation{};
    size_t offset = 0;
    if (!ptr || ld <= 0 || row < 0 || col < 0 || rows <= 0 || cols <= 0 ||
        row > ld - rows || !Find(ptr, &allocation, &offset) || allocation.kind != BackingKind::kRam)
        return false;
    const size_t limit = (allocation.bytes - offset) / sizeof(Complex);
    const uint64_t end = static_cast<uint64_t>(col + cols - 1) * ld + row + rows;
    return end <= limit;
}
// Pack/upload or download/write a rectangle in bounded, pinned column strips.
// The buffer is synchronized before reuse, including the pageable fallback.
Status UploadRect(Complex* device, int64_t device_ld,
                  const Complex* src, int64_t ld, int64_t row, int64_t col,
                  int64_t rows, int64_t cols, HostBuffer& host, cudaStream_t stream,
                  const std::vector<int>* pivots = nullptr)
{
    if (UseDirectPinned() && !pivots && RamRect(src, ld, row, col, rows, cols))
    {
        CHECK(Cuda(cudaMemcpy2DAsync(device, device_ld * sizeof(Complex),
                                    src + col * ld + row, ld * sizeof(Complex),
                                    rows * sizeof(Complex), cols, cudaMemcpyHostToDevice, stream)));
        CHECK(Sync(stream));
        const uint64_t bytes = static_cast<uint64_t>(rows) * cols * sizeof(Complex);
        stats.ram_bytes_read += bytes;
        return Status::kSuccess;
    }
    if (host.count < static_cast<size_t>(rows)) return Status::kInvalidValue;
    const int64_t strip = std::max<int64_t>(1, host.count / rows);
    for (int64_t j = 0; j < cols; j += strip)
    {
        const auto nc = std::min(strip, cols - j);
        CHECK(ReadRect(src, ld, row, col + j, rows, nc, host.p, stream));
        if (pivots)
            for (int64_t k = 0; k < rows; ++k)
                if ((*pivots)[k] != k)
                    for (int64_t c = 0; c < nc; ++c)
                        std::swap(host.p[c * rows + k], host.p[c * rows + (*pivots)[k]]);
        CHECK(Cuda(cudaMemcpy2DAsync(device + j * device_ld, device_ld * sizeof(Complex),
                                    host.p, rows * sizeof(Complex), rows * sizeof(Complex), nc,
                                    cudaMemcpyHostToDevice, stream)));
        CHECK(Sync(stream));
    }
    return Status::kSuccess;
}
Status DownloadRect(const Complex* device, int64_t device_ld,
                    Complex* dst, int64_t ld, int64_t row, int64_t col,
                    int64_t rows, int64_t cols, HostBuffer& host, cudaStream_t stream)
{
    if (UseDirectPinned() && RamRect(dst, ld, row, col, rows, cols))
    {
        CHECK(Cuda(cudaMemcpy2DAsync(dst + col * ld + row, ld * sizeof(Complex),
                                    device, device_ld * sizeof(Complex), rows * sizeof(Complex),
                                    cols, cudaMemcpyDeviceToHost, stream)));
        CHECK(Sync(stream));
        const uint64_t bytes = static_cast<uint64_t>(rows) * cols * sizeof(Complex);
        stats.ram_bytes_written += bytes;
        return Status::kSuccess;
    }
    if (host.count < static_cast<size_t>(rows)) return Status::kInvalidValue;
    const int64_t strip = std::max<int64_t>(1, host.count / rows);
    for (int64_t j = 0; j < cols; j += strip)
    {
        const auto nc = std::min(strip, cols - j);
        CHECK(Cuda(cudaMemcpy2DAsync(host.p, rows * sizeof(Complex),
                                    device + j * device_ld, device_ld * sizeof(Complex),
                                    rows * sizeof(Complex), nc, cudaMemcpyDeviceToHost, stream)));
        CHECK(Sync(stream));
        CHECK(WriteRect(dst, ld, row, col + j, rows, nc, host.p, stream));
    }
    return Status::kSuccess;
}
#if USE_DOUBLE_PRECISION
#define OOC_TRSM cublasZtrsm
#else
#define OOC_TRSM cublasCtrsm
#endif
} // namespace

void Configure(int ox, int oy)
{
    const char* mode = std::getenv("RCWA_MEMORY_MODE");
    policy = BackingPolicy::kFile;
    if (mode && (!std::strcmp(mode, "ram") || !std::strcmp(mode, "ram_only")))
    {
        policy = BackingPolicy::kRamOnly;
        enabled = true;
        return;
    }
    if (mode && !std::strcmp(mode, "hybrid"))
    {
        policy = BackingPolicy::kHybrid;
        enabled = true;
        return;
    }
    if (mode && !std::strcmp(mode, "resident"))
    {
        enabled = false;
        return;
    }
    if (mode && (!std::strcmp(mode, "out_of_core") || !std::strcmp(mode, "ooc")))
    {
        enabled = true;
        return;
    }
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    const long double d = 2.L * (2.L * ox + 1) * (2.L * oy + 1);
    const long double peak = 23.L * d * d * sizeof(Complex);
    enabled = peak > static_cast<long double>(free) * 0.80L;
}
bool Enabled() { return enabled; }
bool RamOnly() { return policy == BackingPolicy::kRamOnly; }
bool IsHost(const void* p)
{
    Allocation a{};
    size_t off = 0;
    return p && Find(p, &a, &off);
}
bool IsFile(const void* p)
{
    Allocation a{};
    size_t off = 0;
    return p && Find(p, &a, &off) && a.kind == BackingKind::kFile;
}
bool IsRam(const void* p)
{
    Allocation a{};
    size_t off = 0;
    return p && Find(p, &a, &off) && a.kind == BackingKind::kRam;
}
Stats GetStats() { return stats; }
int TileSize()
{
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    const size_t budget = std::min(EnvSize("RCWA_GPU_BUDGET_MB", 2048) * 1024 * 1024, free / 2);
    const size_t b =
        static_cast<size_t>(std::sqrt(static_cast<double>(budget) / (3 * sizeof(Complex))));
    return static_cast<int>(
        std::max<size_t>(64, std::min(EnvSize("RCWA_TILE_SIZE", 4096), (b / 64) * 64)));
}
cudaError_t Allocate(Complex** ptr, size_t bytes)
{
    if (!ptr || !bytes) return cudaErrorInvalidValue;
    *ptr = nullptr;
    if (!enabled || bytes < EnvSize("RCWA_FILE_MIN_MB", 8) * 1024 * 1024)
        return cudaMalloc(ptr, bytes);
    const char* residency = std::getenv("RCWA_RESIDENT_MB");
    const size_t limit = residency && !std::strcmp(residency, "0")
                             ? 0
                             : EnvSize("RCWA_RESIDENT_MB", 3072) * 1024 * 1024;
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    {
        std::lock_guard<std::mutex> guard(file_mutex);
        if (limit && resident_bytes <= limit && bytes <= limit - resident_bytes &&
            free_bytes > bytes + std::min<size_t>(free_bytes / 2, 3072ULL * 1024 * 1024))
        {
            const auto status = cudaMalloc(ptr, bytes);
            if (status == cudaSuccess)
            {
                resident[*ptr] = bytes;
                resident_bytes += bytes;
                return status;
            }
            if (status != cudaErrorMemoryAllocation) return status;
            cudaGetLastError();
        }
    }
    if (policy != BackingPolicy::kFile)
    {
        std::lock_guard<std::mutex> guard(file_mutex);
        const size_t limit = RamBudgetBytes();
        const size_t available = AvailableRamBytes();
        const size_t reserve = EnvSize("RCWA_RAM_RESERVE_MB", 1024) * 1024 * 1024;
        const bool fits = stats.ram_bytes <= limit && bytes <= limit - stats.ram_bytes &&
                          available >= reserve && bytes <= available - reserve;
        cudaError_t result = cudaErrorMemoryAllocation;
        if (fits)
        {
            // Driver-pinned host RAM cannot be swapped out. Do not replace this
            // with malloc/tmpfs in ram-only mode: either could use disk swap.
            result = cudaHostAlloc(ptr, bytes, cudaHostAllocDefault);
            if (result == cudaSuccess)
            {
                backing.emplace(reinterpret_cast<uintptr_t>(*ptr),
                                Allocation{*ptr, bytes, -1, BackingKind::kRam});
                stats.ram_bytes += bytes;
                stats.peak_ram_bytes = std::max(stats.peak_ram_bytes, stats.ram_bytes);
                ++stats.ram_allocations;
                return cudaSuccess;
            }
            cudaGetLastError();
        }
        if (policy == BackingPolicy::kRamOnly)
        {
            std::fprintf(stderr,
                         "RCWA RAM-only allocation failed: request=%.1f MiB used=%.1f MiB "
                         "budget=%.1f MiB available=%.1f MiB reserve=%.1f MiB; "
                         "SSD fallback disabled\n",
                         bytes / 1048576.0, stats.ram_bytes / 1048576.0, limit / 1048576.0,
                         available / 1048576.0, reserve / 1048576.0);
            *ptr = nullptr;
            return result;
        }
        *ptr = nullptr;
    }
    const char* root = std::getenv("RCWA_CACHE_DIR");
    std::string dir = root && *root ? root : ".rcwa-cache";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        std::fprintf(stderr, "RCWA cache: %s\n", ec.message().c_str());
        return cudaErrorUnknown;
    }
    struct statvfs fs{};
    if (statvfs(dir.c_str(), &fs) || bytes > static_cast<uint64_t>(fs.f_bavail) * fs.f_frsize)
        return cudaErrorMemoryAllocation;
    std::string path = dir + "/matrix-XXXXXX";
    std::vector<char> name(path.begin(), path.end());
    name.push_back(0);
    const int fd = mkstemp(name.data());
    if (fd < 0) return cudaErrorUnknown;
    unlink(name.data()); // automatic cleanup even if a computation is interrupted
    if (ftruncate(fd, static_cast<off_t>(bytes)))
    {
        close(fd);
        return cudaErrorMemoryAllocation;
    }
    // The mapping supplies stable host addresses for existing submatrix offsets.
    // No mapped pages are touched: all access uses bounded pread/pwrite buffers.
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED)
    {
        close(fd);
        return cudaErrorMemoryAllocation;
    }
    {
        std::lock_guard<std::mutex> guard(file_mutex);
        backing.emplace(reinterpret_cast<uintptr_t>(p), Allocation{p, bytes, fd, BackingKind::kFile});
        stats.file_bytes += bytes;
        stats.peak_file_bytes = std::max(stats.peak_file_bytes, stats.file_bytes);
        ++stats.file_allocations;
    }
    *ptr = static_cast<Complex*>(p);
    return cudaSuccess;
}
cudaError_t Free(void* ptr)
{
    if (!ptr) return cudaSuccess;
    {
        std::lock_guard<std::mutex> guard(file_mutex);
        const auto device = resident.find(ptr);
        if (device != resident.end())
        {
            resident_bytes -= device->second;
            resident.erase(device);
        }
        const auto it = backing.find(reinterpret_cast<uintptr_t>(ptr));
        if (it != backing.end())
        {
            if (it->second.kind == BackingKind::kRam)
            {
                const auto result = cudaFreeHost(it->second.ptr);
                if (result != cudaSuccess) return result;
                stats.ram_bytes -= it->second.bytes;
                backing.erase(it);
                return cudaSuccess;
            }
            munmap(it->second.ptr, it->second.bytes);
            close(it->second.fd);
            stats.file_bytes -= it->second.bytes;
            backing.erase(it);
            return cudaSuccess;
        }
    }
    return cudaFree(ptr);
}
Status Read(const Complex* src, Complex* host, size_t count, cudaStream_t stream)
{
    Allocation a{};
    size_t off = 0;
    if (Find(src, &a, &off))
    {
        if (count > (a.bytes - off) / sizeof(Complex)) return Status::kInvalidValue;
        if (a.kind == BackingKind::kRam)
        {
            std::memmove(host, static_cast<const char*>(a.ptr) + off, count * sizeof(Complex));
            stats.ram_bytes_read += count * sizeof(Complex);
            return Status::kSuccess;
        }
        CHECK(IO(a.fd, host, count * sizeof(Complex), off, false));
        stats.bytes_read += count * sizeof(Complex);
        return Status::kSuccess;
    }
    CHECK(
        Cuda(cudaMemcpyAsync(host, src, count * sizeof(Complex), cudaMemcpyDeviceToHost, stream)));
    return Sync(stream);
}
Status Write(Complex* dst, const Complex* host, size_t count, cudaStream_t stream)
{
    Allocation a{};
    size_t off = 0;
    if (Find(dst, &a, &off))
    {
        if (count > (a.bytes - off) / sizeof(Complex)) return Status::kInvalidValue;
        if (a.kind == BackingKind::kRam)
        {
            std::memmove(static_cast<char*>(a.ptr) + off, host, count * sizeof(Complex));
            stats.ram_bytes_written += count * sizeof(Complex);
            return Status::kSuccess;
        }
        CHECK(IO(a.fd, const_cast<Complex*>(host), count * sizeof(Complex), off, true));
        stats.bytes_written += count * sizeof(Complex);
        return Status::kSuccess;
    }
    CHECK(
        Cuda(cudaMemcpyAsync(dst, host, count * sizeof(Complex), cudaMemcpyHostToDevice, stream)));
    return Sync(stream);
}
Status ReadRect(const Complex* src,
                int64_t ld,
                int64_t row,
                int64_t col,
                int64_t rows,
                int64_t cols,
                Complex* host,
                cudaStream_t stream)
{
    if (!src || !host || rows <= 0 || cols <= 0 || row < 0 || col < 0 || row + rows > ld)
        return Status::kInvalidValue;
    if (row == 0 && rows == ld) return Read(src + col * ld, host, rows * cols, stream);
    if (!IsHost(src))
    {
        CHECK(Cuda(cudaMemcpy2DAsync(host,
                                     rows * sizeof(Complex),
                                     src + col * ld + row,
                                     ld * sizeof(Complex),
                                     rows * sizeof(Complex),
                                     cols,
                                     cudaMemcpyDeviceToHost,
                                     stream)));
        return Sync(stream);
    }
    for (int64_t j = 0; j < cols; ++j)
        CHECK(Read(src + (col + j) * ld + row, host + j * rows, rows, stream));
    return Status::kSuccess;
}
Status WriteRect(Complex* dst,
                 int64_t ld,
                 int64_t row,
                 int64_t col,
                 int64_t rows,
                 int64_t cols,
                 const Complex* host,
                 cudaStream_t stream)
{
    if (!dst || !host || rows <= 0 || cols <= 0 || row < 0 || col < 0 || row + rows > ld)
        return Status::kInvalidValue;
    if (row == 0 && rows == ld)
    {
        CHECK(Write(dst + col * ld, host, rows * cols, stream));
        return Flush(dst);
    }
    if (!IsHost(dst))
    {
        CHECK(Cuda(cudaMemcpy2DAsync(dst + col * ld + row,
                                     ld * sizeof(Complex),
                                     host,
                                     rows * sizeof(Complex),
                                     rows * sizeof(Complex),
                                     cols,
                                     cudaMemcpyHostToDevice,
                                     stream)));
        return Sync(stream);
    }
    for (int64_t j = 0; j < cols; ++j)
        CHECK(Write(dst + (col + j) * ld + row, host + j * rows, rows, stream));
    return Flush(dst);
}
cudaError_t
MemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream)
{
    const bool sf = IsHost(src), df = IsHost(dst);
    if (!sf && !df) return cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (bytes % sizeof(Complex)) return cudaErrorInvalidValue;
    if (Sync(stream) != Status::kSuccess) return cudaErrorUnknown;
    const size_t chunk = 1024 * 1024, count = bytes / sizeof(Complex);
    std::vector<Complex> buf(std::min(chunk, count));
    for (size_t i = 0; i < count; i += chunk)
    {
        size_t c = std::min(chunk, count - i);
        if (!sf && kind == cudaMemcpyHostToDevice)
            std::memcpy(buf.data(), static_cast<const Complex*>(src) + i, c * sizeof(Complex));
        else if (Read(static_cast<const Complex*>(src) + i, buf.data(), c, stream) !=
                 Status::kSuccess)
            return cudaErrorUnknown;
        if (!df && kind == cudaMemcpyDeviceToHost)
            std::memcpy(static_cast<Complex*>(dst) + i, buf.data(), c * sizeof(Complex));
        else if (Write(static_cast<Complex*>(dst) + i, buf.data(), c, stream) != Status::kSuccess)
            return cudaErrorUnknown;
    }
    return Flush(dst) == Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}
cudaError_t MemsetAsync(void* dst, int value, size_t bytes, cudaStream_t stream)
{
    if (!IsHost(dst)) return cudaMemsetAsync(dst, value, bytes, stream);
    if (Sync(stream) != Status::kSuccess) return cudaErrorUnknown;
    Allocation a{};
    size_t off = 0;
    Find(dst, &a, &off);
    if (bytes > a.bytes - off) return cudaErrorInvalidValue;
    if (a.kind == BackingKind::kRam)
    {
        std::memset(static_cast<char*>(a.ptr) + off, value, bytes);
        return cudaSuccess;
    }
    if (policy == BackingPolicy::kRamOnly) return cudaErrorInvalidValue;
    if (!value && off == 0 && bytes == a.bytes)
    {
        if (ftruncate(a.fd, 0) || ftruncate(a.fd, a.bytes)) return cudaErrorUnknown;
        return cudaSuccess;
    }
    std::vector<char> zero(std::min<size_t>(bytes, 8 * 1024 * 1024), static_cast<char>(value));
    for (size_t i = 0; i < bytes; i += zero.size())
        if (IO(a.fd, zero.data(), std::min(zero.size(), bytes - i), off + i, true) !=
            Status::kSuccess)
            return cudaErrorUnknown;
    return Flush(dst) == Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

Status Gemm(cublasHandle_t handle,
            const Complex* A,
            int64_t lda,
            const Complex* B,
            int64_t ldb,
            Complex* C,
            int64_t ldc,
            int64_t m,
            int64_t n,
            int64_t k,
            cublasOperation_t ta,
            cublasOperation_t tb,
            Complex alpha,
            Complex beta,
            cudaStream_t stream)
{
    if (!A || !B || !C || m <= 0 || n <= 0 || k <= 0 || A == C || B == C ||
        m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max() || lda < (ta == CUBLAS_OP_N ? m : k) ||
        ldb < (tb == CUBLAS_OP_N ? k : n) || ldc < m)
        return Status::kInvalidValue;
    CHECK(Sync(stream));
    ++stats.gemms;
    const int64_t b = TileSize(), bm = std::min(b, m), bn = std::min(b, n), bk = std::min(b, k);
    const bool af = IsHost(A), bf = IsHost(B), cf = IsHost(C);
    size_t free_bytes = 0, total_bytes = 0;
    CHECK(Cuda(cudaMemGetInfo(&free_bytes, &total_bytes)));
    const size_t capacity = std::min(EnvSize("RCWA_GPU_BUDGET_MB", 2048) * 1024 * 1024,
                                     free_bytes / 2) / sizeof(Complex);
    const size_t csize = cf ? bm * bn : 0, abase = af ? bm * bk : 0;
    // Keep a whole B column band across all row tiles. If the budget allows,
    // also upload A once for this operation instead of once per output tile.
    const char* reuse_env = std::getenv("RCWA_GEMM_REUSE");
    const bool reuse = !reuse_env || std::strcmp(reuse_env, "0") != 0;
    const bool cache_b = reuse && bf && capacity >= abase + csize &&
                         static_cast<size_t>(k) <= (capacity - abase - csize) / bn;
    const size_t bsize = bf ? (cache_b ? k * bn : bk * bn) : 0;
    const bool cache_a = reuse && af && capacity >= bsize + csize &&
                         static_cast<size_t>(m) <= (capacity - bsize - csize) / k;
    const size_t asize = af ? (cache_a ? m * k : bm * bk) : 0;
    const bool stage_a_needed = af && !cache_a, stage_b_needed = bf && !cache_b;
    const bool direct_a = stage_a_needed && UseDirectPinned() && IsRam(A);
    const bool direct_b = stage_b_needed && UseDirectPinned() && IsRam(B);
    const size_t extra = (stage_a_needed ? static_cast<size_t>(bm) * bk : 0) +
                         (stage_b_needed ? static_cast<size_t>(bk) * bn : 0);
    const char* pipeline_env = std::getenv("RCWA_GEMM_PIPELINE");
    bool pipeline = (!pipeline_env || std::strcmp(pipeline_env, "0") != 0) &&
                    (stage_a_needed || stage_b_needed) && k > bk &&
                    asize + bsize + csize <= capacity &&
                    extra <= capacity - asize - bsize - csize &&
                    std::max(stage_a_needed && !direct_a ? static_cast<size_t>(bm) * bk : 0,
                             stage_b_needed && !direct_b ? static_cast<size_t>(bk) * bn : 0) <=
                        EnvSize("RCWA_HOST_BUFFER_MB", 128) * 1024 * 1024 / sizeof(Complex);
    Device da, db, dc, da_alt, db_alt;
    if (af) CHECK(da.Alloc(asize));
    if (bf) CHECK(db.Alloc(bsize));
    if (cf) CHECK(dc.Alloc(csize));
    std::array<HostBuffer, 2> stage_a, stage_b;
    std::array<CopyEvent, 2> ready, used;
    CopyStream copy_stream;
    bool slot_ready[2] = {false, false}, slot_used[2] = {false, false};
    if (pipeline)
    {
        for (int s = 0; s < 2; ++s)
        {
            if (stage_a_needed && !direct_a &&
                stage_a[s].Alloc(static_cast<size_t>(bm) * bk) != Status::kSuccess) pipeline = false;
            if (pipeline && stage_b_needed && !direct_b &&
                stage_b[s].Alloc(static_cast<size_t>(bk) * bn) != Status::kSuccess) pipeline = false;
            if ((stage_a_needed && !direct_a && !stage_a[s].pinned) ||
                (stage_b_needed && !direct_b && !stage_b[s].pinned)) pipeline = false;
            if (!pipeline) break;
        }
        if (pipeline && stage_a_needed && da_alt.Alloc(static_cast<size_t>(bm) * bk) != Status::kSuccess)
        { cudaGetLastError(); pipeline = false; }
        if (pipeline && stage_b_needed && db_alt.Alloc(static_cast<size_t>(bk) * bn) != Status::kSuccess)
        { cudaGetLastError(); pipeline = false; }
        if (pipeline)
        {
            if (cudaStreamCreateWithFlags(&copy_stream.stream, cudaStreamNonBlocking) != cudaSuccess)
            { cudaGetLastError(); pipeline = false; }
            for (int s = 0; pipeline && s < 2; ++s)
                if (ready[s].Create() != Status::kSuccess || used[s].Create() != Status::kSuccess)
                { cudaGetLastError(); pipeline = false; }
        }
        if (!pipeline)
        {
            da_alt.Reset(); db_alt.Reset();
            for (int s = 0; s < 2; ++s) { stage_a[s].Reset(); stage_b[s].Reset(); }
        }
    }
    HostBuffer host;
    CHECK(host.Alloc(std::max({asize, bsize, csize})));
    Fence compute_fence{stream};
    if (cache_a)
    {
        const auto rows = ta == CUBLAS_OP_N ? m : k;
        const auto cols = ta == CUBLAS_OP_N ? k : m;
        CHECK(UploadRect(da.p, rows, A, lda, 0, 0, rows, cols, host, stream));
    }
    const Complex one = make_complex(1, 0);
    for (int64_t j = 0; j < n; j += bn)
    {
        const int nj = static_cast<int>(std::min(bn, n - j));
        if (cache_b)
        {
            const auto rows = tb == CUBLAS_OP_N ? k : nj;
            const auto cols = tb == CUBLAS_OP_N ? nj : k;
            CHECK(UploadRect(db.p, rows, B, ldb, tb == CUBLAS_OP_N ? 0 : j,
                             tb == CUBLAS_OP_N ? j : 0, rows, cols, host, stream));
        }
        for (int64_t i = 0; i < m; i += bm)
        {
            const int mi = static_cast<int>(std::min(bm, m - i));
            Complex* cp = cf ? dc.p : C + i + j * ldc;
            const int cld = cf ? mi : static_cast<int>(ldc);
            if (cf && (beta.x != 0 || beta.y != 0))
                CHECK(UploadRect(dc.p, mi, C, ldc, i, j, mi, nj, host, stream));
            if (pipeline)
            {
                const auto stage = [&](int slot, int64_t t) -> Status {
                    if (slot_ready[slot]) CHECK(Cuda(cudaEventSynchronize(ready[slot].event)));
                    if (slot_used[slot])
                        CHECK(Cuda(cudaStreamWaitEvent(copy_stream.stream, used[slot].event, 0)));
                    const int kt = static_cast<int>(std::min(bk, k - t));
                    const int ar = ta == CUBLAS_OP_N ? mi : kt, ac = ta == CUBLAS_OP_N ? kt : mi;
                    const int br = tb == CUBLAS_OP_N ? kt : nj, bc = tb == CUBLAS_OP_N ? nj : kt;
                    if (stage_a_needed)
                    {
                        const int64_t row = ta == CUBLAS_OP_N ? i : t;
                        const int64_t col = ta == CUBLAS_OP_N ? t : i;
                        const Complex* source = stage_a[slot].p;
                        size_t pitch = static_cast<size_t>(ar) * sizeof(Complex);
                        if (direct_a)
                        {
                            if (!RamRect(A, lda, row, col, ar, ac)) return Status::kInvalidValue;
                            source = A + col * lda + row;
                            pitch = static_cast<size_t>(lda) * sizeof(Complex);
                            const uint64_t bytes = static_cast<uint64_t>(ar) * ac * sizeof(Complex);
                            stats.ram_bytes_read += bytes;
                        }
                        else CHECK(ReadRect(A, lda, row, col, ar, ac, stage_a[slot].p, stream));
                        CHECK(Cuda(cudaMemcpy2DAsync(slot ? da_alt.p : da.p, ar * sizeof(Complex),
                                                     source, pitch, ar * sizeof(Complex), ac,
                                                     cudaMemcpyHostToDevice, copy_stream.stream)));
                    }
                    if (stage_b_needed)
                    {
                        const int64_t row = tb == CUBLAS_OP_N ? t : j;
                        const int64_t col = tb == CUBLAS_OP_N ? j : t;
                        const Complex* source = stage_b[slot].p;
                        size_t pitch = static_cast<size_t>(br) * sizeof(Complex);
                        if (direct_b)
                        {
                            if (!RamRect(B, ldb, row, col, br, bc)) return Status::kInvalidValue;
                            source = B + col * ldb + row;
                            pitch = static_cast<size_t>(ldb) * sizeof(Complex);
                            const uint64_t bytes = static_cast<uint64_t>(br) * bc * sizeof(Complex);
                            stats.ram_bytes_read += bytes;
                        }
                        else CHECK(ReadRect(B, ldb, row, col, br, bc, stage_b[slot].p, stream));
                        CHECK(Cuda(cudaMemcpy2DAsync(slot ? db_alt.p : db.p, br * sizeof(Complex),
                                                     source, pitch, br * sizeof(Complex), bc,
                                                     cudaMemcpyHostToDevice, copy_stream.stream)));
                    }
                    CHECK(Cuda(cudaEventRecord(ready[slot].event, copy_stream.stream)));
                    slot_ready[slot] = true;
                    return Status::kSuccess;
                };
                CHECK(stage(0, 0));
                for (int64_t t = 0, tile = 0; t < k; t += bk, ++tile)
                {
                    const int slot = static_cast<int>(tile & 1);
                    const int kt = static_cast<int>(std::min(bk, k - t));
                    const int ar = ta == CUBLAS_OP_N ? mi : kt;
                    const int br = tb == CUBLAS_OP_N ? kt : nj;
                    CHECK(Cuda(cudaStreamWaitEvent(stream, ready[slot].event, 0)));
                    const Complex* ap = A + (ta == CUBLAS_OP_N ? i + t * lda : t + i * lda);
                    int ald = static_cast<int>(lda);
                    if (cache_a)
                    {
                        ald = static_cast<int>(ta == CUBLAS_OP_N ? m : k);
                        ap = da.p + (ta == CUBLAS_OP_N ? i + t * m : t + i * k);
                    }
                    else if (stage_a_needed) { ap = slot ? da_alt.p : da.p; ald = ar; }
                    const Complex* bp = B + (tb == CUBLAS_OP_N ? t + j * ldb : j + t * ldb);
                    int bld = static_cast<int>(ldb);
                    if (cache_b)
                    {
                        bld = static_cast<int>(tb == CUBLAS_OP_N ? k : nj);
                        bp = db.p + (tb == CUBLAS_OP_N ? t : t * nj);
                    }
                    else if (stage_b_needed) { bp = slot ? db_alt.p : db.p; bld = br; }
                    const Complex bt = t == 0 ? beta : one;
                    CHECK(Blas(CUBLAS_GEMM(handle, ta, tb, mi, nj, kt, &alpha,
                                           ap, ald, bp, bld, &bt, cp, cld)));
                    CHECK(Cuda(cudaEventRecord(used[slot].event, stream)));
                    slot_used[slot] = true;
                    if (t + bk < k) CHECK(stage(1 - slot, t + bk));
                }
                if (cf) CHECK(DownloadRect(dc.p, mi, C, ldc, i, j, mi, nj, host, stream));
                continue;
            }
            for (int64_t t = 0; t < k; t += bk)
            {
                const int kt = static_cast<int>(std::min(bk, k - t));
                const int ar = ta == CUBLAS_OP_N ? mi : kt, ac = ta == CUBLAS_OP_N ? kt : mi;
                const Complex* ap = A + (ta == CUBLAS_OP_N ? i + t * lda : t + i * lda);
                int ald = static_cast<int>(lda);
                if (cache_a)
                {
                    ald = static_cast<int>(ta == CUBLAS_OP_N ? m : k);
                    ap = da.p + (ta == CUBLAS_OP_N ? i + t * m : t + i * k);
                }
                else if (af)
                {
                    CHECK(UploadRect(da.p, ar, A, lda, ta == CUBLAS_OP_N ? i : t,
                                     ta == CUBLAS_OP_N ? t : i, ar, ac, host, stream));
                    ap = da.p;
                    ald = ar;
                }
                const int br = tb == CUBLAS_OP_N ? kt : nj, bc = tb == CUBLAS_OP_N ? nj : kt;
                const Complex* bp = B + (tb == CUBLAS_OP_N ? t + j * ldb : j + t * ldb);
                int bld = static_cast<int>(ldb);
                if (cache_b)
                {
                    bld = static_cast<int>(tb == CUBLAS_OP_N ? k : nj);
                    bp = db.p + (tb == CUBLAS_OP_N ? t : t * nj);
                }
                else if (bf)
                {
                    CHECK(UploadRect(db.p, br, B, ldb, tb == CUBLAS_OP_N ? t : j,
                                     tb == CUBLAS_OP_N ? j : t, br, bc, host, stream));
                    bp = db.p;
                    bld = br;
                }
                const Complex bt = t == 0 ? beta : one;
                CHECK(Blas(CUBLAS_GEMM(handle, ta, tb, mi, nj, kt, &alpha,
                                       ap, ald, bp, bld, &bt, cp, cld)));
            }
            if (cf) CHECK(DownloadRect(dc.p, mi, C, ldc, i, j, mi, nj, host, stream));
        }
    }
    CHECK(Sync(stream));
    return Status::kSuccess;
}

Status Elementwise(ElementOp op,
                   const Complex* A,
                   const Complex* B,
                   Complex* C,
                   int64_t rows,
                   int64_t cols,
                   Complex alpha,
                   cudaStream_t stream)
{
    if (!A || !C || rows <= 0 || cols <= 0 || rows > std::numeric_limits<int64_t>::max() / cols ||
        ((op == ElementOp::Add || op == ElementOp::Sub || op == ElementOp::ScaleAdd ||
          op == ElementOp::LeftDiag || op == ElementOp::RightDiag) &&
         !B))
        return Status::kInvalidValue;
    CHECK(Sync(stream));
    const int64_t count = rows * cols, chunk = std::min<int64_t>(count, 1024 * 1024);
    std::vector<Complex> a(chunk), b(chunk), diag;
    if (op == ElementOp::LeftDiag || op == ElementOp::RightDiag)
    {
        diag.resize(op == ElementOp::LeftDiag ? rows : cols);
        CHECK(Read(A, diag.data(), diag.size(), stream));
    }
    for (int64_t off = 0; off < count; off += chunk)
    {
        const auto c = std::min(chunk, count - off);
        if (diag.empty()) CHECK(Read(A + off, a.data(), c, stream));
        if (B) CHECK(Read(B + off, b.data(), c, stream));
        for (int64_t x = 0; x < c; ++x) switch (op)
            {
            case ElementOp::Add:
                a[x] = ComplexAdd(a[x], b[x]);
                break;
            case ElementOp::Sub:
                a[x] = ComplexSub(a[x], b[x]);
                break;
            case ElementOp::Scale:
                a[x] = ComplexMul(a[x], alpha);
                break;
            case ElementOp::ScaleAdd:
                a[x] = ComplexAdd(a[x], ComplexMul(b[x], alpha));
                break;
            case ElementOp::LeftDiag:
                a[x] = ComplexMul(diag[(off + x) % rows], b[x]);
                break;
            case ElementOp::RightDiag:
                a[x] = ComplexMul(b[x], diag[(off + x) / rows]);
                break;
            case ElementOp::Inverse:
                a[x] = ComplexInv(a[x]);
                break;
            }
        CHECK(Write(C + off, a.data(), c, stream));
        if (IsHost(C)) CHECK(Flush(C));
    }
    return Status::kSuccess;
}
Status Diagonal(Complex* A,
                int64_t ld,
                int64_t row,
                int64_t col,
                int64_t n,
                const Complex* diag,
                Complex value,
                bool add,
                cudaStream_t stream)
{
    if (!A || ld <= 0 || n <= 0 || row < 0 || col < 0 || row > ld - n) return Status::kInvalidValue;
    CHECK(Sync(stream));
    std::vector<Complex> d(n);
    if (diag) CHECK(Read(diag, d.data(), n, stream));
    if (IsHost(A))
    {
        for (int64_t i = 0; i < n; ++i)
        {
            Complex v = diag ? ComplexMul(d[i], value) : value;
            Complex* out = A + (col + i) * ld + row + i;
            if (add)
            {
                Complex old;
                CHECK(Read(out, &old, 1, stream));
                v = ComplexAdd(old, v);
            }
            CHECK(Write(out, &v, 1, stream));
        }
        return Flush(A);
    }
    for (int64_t i = 0; i < n; ++i)
    {
        Complex v = diag ? ComplexMul(d[i], value) : value, old;
        Complex* out = A + (col + i) * ld + row + i;
        if (add)
        {
            CHECK(Read(out, &old, 1, stream));
            v = ComplexAdd(old, v);
        }
        CHECK(Write(out, &v, 1, stream));
    }
    return Status::kSuccess;
}
Status Block(const Complex* src,
             int64_t n,
             Complex* dst,
             int64_t ld,
             int64_t row,
             int64_t col,
             Complex alpha,
             bool add,
             cudaStream_t stream)
{
    if (!src || !dst || n <= 0 || ld <= 0 || row < 0 || col < 0 || row > ld - n)
        return Status::kInvalidValue;
    CHECK(Sync(stream));
    const int64_t b = std::min<int64_t>(1024, n);
    std::vector<Complex> a(b * b), c(add ? b * b : 0);
    for (int64_t j = 0; j < n; j += b)
        for (int64_t i = 0; i < n; i += b)
        {
            const auto nr = std::min(b, n - i), nc = std::min(b, n - j);
            CHECK(ReadRect(src, n, i, j, nr, nc, a.data(), stream));
            if (add) CHECK(ReadRect(dst, ld, row + i, col + j, nr, nc, c.data(), stream));
            for (int64_t x = 0; x < nr * nc; ++x)
            {
                a[x] = ComplexMul(a[x], alpha);
                if (add) a[x] = ComplexAdd(a[x], c[x]);
            }
            CHECK(WriteRect(dst, ld, row + i, col + j, nr, nc, a.data(), stream));
        }
    return Status::kSuccess;
}
Status Norm1(const Complex* A, int64_t n, Complex offset, double* norm, cudaStream_t stream)
{
    if (!A || !norm || n <= 0) return Status::kInvalidValue;
    CHECK(Sync(stream));
    *norm = 0;
    const auto nc = std::max<int64_t>(1, std::min<int64_t>(n, 1024 * 1024 / n));
    std::vector<Complex> a(n * nc);
    for (int64_t j = 0; j < n; j += nc)
    {
        const auto c = std::min(nc, n - j);
        CHECK(Read(A + j * n, a.data(), n * c, stream));
        for (int64_t col = 0; col < c; ++col)
        {
            double sum = 0;
            for (int64_t i = 0; i < n; ++i)
            {
                Complex v = a[col * n + i];
                if (i == j + col) v = ComplexSub(v, offset);
                sum += std::hypot(static_cast<double>(v.x), static_cast<double>(v.y));
            }
            *norm = std::max(*norm, sum);
        }
    }
    return Status::kSuccess;
}
Status Transpose(const Complex* A, Complex* C, int64_t n, cudaStream_t stream)
{
    if (!A || !C || A == C || n <= 0) return Status::kInvalidValue;
    const int64_t b = std::min<int64_t>(TileSize(), n);
    std::vector<Complex> a(b * b), c(b * b);
    for (int64_t j = 0; j < n; j += b)
        for (int64_t i = 0; i < n; i += b)
        {
            auto nr = std::min(b, n - i), nc = std::min(b, n - j);
            CHECK(ReadRect(A, n, i, j, nr, nc, a.data(), stream));
            for (int64_t x = 0; x < nr; ++x)
                for (int64_t y = 0; y < nc; ++y) c[x * nc + y] = a[y * nr + x];
            CHECK(WriteRect(C, n, j, i, nc, nr, c.data(), stream));
        }
    return Status::kSuccess;
}

Status Solve(cublasHandle_t blas,
             cusolverDnHandle_t solver,
             const Complex* A,
             const Complex* B,
             Complex* X,
             int64_t n,
             int64_t nrhs,
             cublasOperation_t trans,
             cudaStream_t stream)
{
    if (!A || !B || !X || n <= 0 || nrhs <= 0 || n > std::numeric_limits<int>::max())
        return Status::kInvalidValue;
    CHECK(Sync(stream));
    ++stats.solves;
    struct Matrix
    {
        Complex* p = nullptr;
        ~Matrix() { Free(p); }
    } lu;
    const bool old = enabled;
    enabled = true;
    const auto alloc = Allocate(&lu.p, static_cast<size_t>(n) * n * sizeof(Complex));
    enabled = old;
    CHECK(Cuda(alloc));
    if (trans != CUBLAS_OP_N)
    {
        CHECK(Transpose(A, lu.p, n, stream));
        if (trans == CUBLAS_OP_C)
        {
            std::vector<Complex> buf(std::min<int64_t>(n * n, 1024 * 1024));
            for (int64_t i = 0; i < n * n; i += buf.size())
            {
                auto c = std::min<int64_t>(buf.size(), n * n - i);
                CHECK(Read(lu.p + i, buf.data(), c, stream));
                for (int64_t j = 0; j < c; ++j) buf[j].y = -buf[j].y;
                CHECK(Write(lu.p + i, buf.data(), c, stream));
            }
            CHECK(Flush(lu.p));
        }
        return Solve(blas, solver, lu.p, B, X, n, nrhs, CUBLAS_OP_N, stream);
    }
    CHECK(Cuda(MemcpyAsync(
        lu.p, A, static_cast<size_t>(n) * n * sizeof(Complex), cudaMemcpyDeviceToDevice, stream)));
    CHECK(Sync(stream));
    const size_t host_elements = EnvSize("RCWA_HOST_BUFFER_MB", 128) * 1024 * 1024 /
                                 sizeof(Complex);
    if (host_elements < static_cast<size_t>(n)) return Status::kInvalidValue;
    const int64_t panel = std::min<int64_t>({n, static_cast<int64_t>(EnvSize("RCWA_LU_PANEL", 512)),
                                            static_cast<int64_t>(host_elements / n)});
    Device dpanel, drhs;
    CHECK(dpanel.Alloc(n * panel));
    CHECK(drhs.Alloc(panel * panel));
    struct Ints
    {
        int *p = nullptr, *info = nullptr;
        ~Ints()
        {
            cudaFree(p);
            cudaFree(info);
        }
    } dev;
    CHECK(Cuda(cudaMalloc(&dev.p, panel * sizeof(int))));
    CHECK(Cuda(cudaMalloc(&dev.info, sizeof(int))));
    HostBuffer host;
    CHECK(host.Alloc(n * panel));
    std::vector<int> ipiv(panel), pivots(n);
    const Complex one = make_complex(1, 0), minus = make_complex(-1, 0);
    for (int64_t k = 0; k < n; k += panel)
    {
        const int rows = static_cast<int>(n - k), nb = static_cast<int>(std::min(panel, n - k));
        CHECK(ReadRect(lu.p, n, k, k, rows, nb, host.data(), stream));
        CHECK(Cuda(cudaMemcpyAsync(dpanel.p,
                                   host.data(),
                                   static_cast<size_t>(rows) * nb * sizeof(Complex),
                                   cudaMemcpyHostToDevice,
                                   stream)));
        int lwork = 0;
        if (CUSOLVER_GETRF_BUFFER_SIZE(solver, rows, nb, dpanel.p, rows, &lwork) !=
            CUSOLVER_STATUS_SUCCESS)
            return Status::kCusolverError;
        Device workspace;
        CHECK(workspace.Alloc(std::max(1, lwork)));
        if (CUSOLVER_GETRF(solver, rows, nb, dpanel.p, rows, workspace.p, dev.p, dev.info) !=
            CUSOLVER_STATUS_SUCCESS)
            return Status::kCusolverError;
        int info = 0;
        CHECK(Cuda(cudaMemcpyAsync(&info, dev.info, sizeof(int), cudaMemcpyDeviceToHost, stream)));
        CHECK(Cuda(
            cudaMemcpyAsync(ipiv.data(), dev.p, nb * sizeof(int), cudaMemcpyDeviceToHost, stream)));
        CHECK(Cuda(cudaMemcpyAsync(host.data(),
                                   dpanel.p,
                                   static_cast<size_t>(rows) * nb * sizeof(Complex),
                                   cudaMemcpyDeviceToHost,
                                   stream)));
        CHECK(Sync(stream));
        if (info != 0)
        {
            std::fprintf(stderr,
                         "RCWA out-of-core LU: panel %lld info=%d\n",
                         static_cast<long long>(k),
                         info);
            return Status::kCusolverError;
        }
        CHECK(WriteRect(lu.p, n, k, k, rows, nb, host.data(), stream));
        for (int t = 0; t < nb; ++t) pivots[k + t] = static_cast<int>(k) + ipiv[t] - 1;
        // Apply panel pivots to previous L columns AND trailing columns.
        for (int64_t j = 0; j < n; j += panel)
        {
            if (j == k) continue;
            const auto nc = std::min(panel, n - j);
            CHECK(ReadRect(lu.p, n, 0, j, n, nc, host.data(), stream));
            for (int t = 0; t < nb; ++t)
                if (pivots[k + t] != k + t)
                    for (int64_t c = 0; c < nc; ++c)
                        std::swap(host[c * n + k + t], host[c * n + pivots[k + t]]);
            CHECK(WriteRect(lu.p, n, 0, j, n, nc, host.data(), stream));
        }
        // Unit-lower solve for U12; the diagonal panel has leading dimension rows.
        for (int64_t j = k + nb; j < n; j += panel)
        {
            const int nc = static_cast<int>(std::min(panel, n - j));
            CHECK(ReadRect(lu.p, n, k, j, nb, nc, host.data(), stream));
            CHECK(Cuda(cudaMemcpyAsync(
                drhs.p, host.data(), nb * nc * sizeof(Complex), cudaMemcpyHostToDevice, stream)));
            CHECK(Blas(OOC_TRSM(blas,
                                CUBLAS_SIDE_LEFT,
                                CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_N,
                                CUBLAS_DIAG_UNIT,
                                nb,
                                nc,
                                &one,
                                dpanel.p,
                                rows,
                                drhs.p,
                                nb)));
            CHECK(Cuda(cudaMemcpyAsync(
                host.data(), drhs.p, nb * nc * sizeof(Complex), cudaMemcpyDeviceToHost, stream)));
            CHECK(Sync(stream));
            CHECK(WriteRect(lu.p, n, k, j, nb, nc, host.data(), stream));
        }
        if (k + nb < n)
            CHECK(Gemm(blas,
                       lu.p + (k + nb) + k * n,
                       n,
                       lu.p + k + (k + nb) * n,
                       n,
                       lu.p + (k + nb) + (k + nb) * n,
                       n,
                       n - k - nb,
                       n - k - nb,
                       nb,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       minus,
                       one,
                       stream));
    }
    // Solve one whole RHS column band on the GPU. The old panel-major loop
    // read/wrote the entire remaining RHS after each 512-row update. Here X
    // stays on the device through BOTH triangular solves and is written once.
    drhs.Reset();
    size_t free_bytes = 0, total_bytes = 0;
    CHECK(Cuda(cudaMemGetInfo(&free_bytes, &total_bytes)));
    const size_t capacity = std::min(EnvSize("RCWA_GPU_BUDGET_MB", 2048) * 1024 * 1024,
                                     free_bytes / 2) / sizeof(Complex);
    const size_t min_rhs = std::min<int64_t>({nrhs, TileSize(), 512});
    const bool resident_lu = !IsHost(lu.p);
    const bool cache_lu = resident_lu || (static_cast<size_t>(n) <= capacity / n &&
                          capacity - static_cast<size_t>(n) * n >= n * min_rhs);
    const size_t factor_elements = cache_lu && !resident_lu ? static_cast<size_t>(n) * n : 0;
    const int64_t rhs_block = std::min<int64_t>({nrhs, TileSize(),
                                static_cast<int64_t>((capacity - factor_elements) / n)});
    if (rhs_block < 1) return Status::kCudaError;
    Device factor, rhs_device;
    if (factor_elements) CHECK(factor.Alloc(factor_elements));
    CHECK(rhs_device.Alloc(n * rhs_block));
    host.Reset();
    CHECK(host.Alloc(std::max<int64_t>(n * rhs_block, n * panel)));
    if (cache_lu)
    {
        if (!resident_lu) CHECK(UploadRect(factor.p, n, lu.p, n, 0, 0, n, n, host, stream));
        ++stats.cached_lu_solves;
    }
    const Complex* full_factor = resident_lu ? lu.p : factor.p;
    for (int64_t j = 0; j < nrhs; j += rhs_block)
    {
        const int nc = static_cast<int>(std::min(rhs_block, nrhs - j));
        CHECK(UploadRect(rhs_device.p, n, B, n, 0, j, n, nc, host, stream, &pivots));
        if (cache_lu)
        {
            CHECK(Blas(OOC_TRSM(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_N, CUBLAS_DIAG_UNIT, static_cast<int>(n), nc,
                                &one, full_factor, static_cast<int>(n), rhs_device.p,
                                static_cast<int>(n))));
            CHECK(Blas(OOC_TRSM(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, static_cast<int>(n), nc,
                                &one, full_factor, static_cast<int>(n), rhs_device.p,
                                static_cast<int>(n))));
        }
        else
        {
            // Bounded fallback for matrices whose complete LU exceeds the
            // workspace budget. Reuse the same column panel for TRSM + GEMM.
            const int64_t blocks = (n + panel - 1) / panel;
            for (int phase = 0; phase < 2; ++phase)
                for (int64_t step = 0; step < blocks; ++step)
                {
                    const int64_t k = (phase == 0 ? step : blocks - 1 - step) * panel;
                    const int nb = static_cast<int>(std::min(panel, n - k));
                    CHECK(UploadRect(dpanel.p, n, lu.p, n, 0, k, n, nb, host, stream));
                    CHECK(Blas(OOC_TRSM(blas, CUBLAS_SIDE_LEFT,
                                        phase == 0 ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                        CUBLAS_OP_N,
                                        phase == 0 ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                                        nb, nc, &one, dpanel.p + k, static_cast<int>(n),
                                        rhs_device.p + k, static_cast<int>(n))));
                    const int64_t row = phase == 0 ? k + nb : 0;
                    const int count = static_cast<int>(phase == 0 ? n - k - nb : k);
                    if (count)
                        CHECK(Blas(CUBLAS_GEMM(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                               count, nc, nb, &minus, dpanel.p + row,
                                               static_cast<int>(n), rhs_device.p + k,
                                               static_cast<int>(n), &one, rhs_device.p + row,
                                               static_cast<int>(n))));
                }
        }
        CHECK(DownloadRect(rhs_device.p, n, X, n, 0, j, n, nc, host, stream));
        ++stats.rhs_slabs;
    }
    return Status::kSuccess;
}
} // namespace rcwa_memory
