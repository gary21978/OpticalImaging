#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "rcwa.h"

namespace
{
using Clock = std::chrono::steady_clock;

double ElapsedMillis(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string FormatComplexCompact(const Complex& value)
{
    double real = static_cast<double>(ComplexReal(value));
    double imag = static_cast<double>(ComplexImag(value));
    const double eps = static_cast<double>(std::numeric_limits<Real>::epsilon()) * 10.0;

    if (std::abs(real) <= eps)
    {
        real = 0.0;
    }
    if (std::abs(imag) <= eps)
    {
        imag = 0.0;
    }

    std::ostringstream oss;
    oss << std::setprecision(6);

    if (imag == 0.0)
    {
        oss << real;
        return oss.str();
    }
    if (real == 0.0)
    {
        oss << imag << "i";
        return oss.str();
    }

    oss << real;
    if (imag > 0.0)
    {
        oss << "+";
    }
    oss << imag << "i";
    return oss.str();
}

void PrintTopLeft2x2Compact(const char* name, const Complex* device_matrix, int n)
{
    const size_t element_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    std::vector<Complex> host_matrix(element_count);
    const cudaError_t copy_status = cudaMemcpy(host_matrix.data(),
                                               device_matrix,
                                               element_count * sizeof(Complex),
                                               cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
    {
        std::cerr << "ERROR: failed to copy " << name << ": "
                  << cudaGetErrorString(copy_status) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << name << " =\n";
    constexpr int limit = 2;
    std::vector<std::string> formatted(limit * limit);
    std::vector<size_t> widths(limit, 0);

    for (int row = 0; row < limit; ++row)
    {
        for (int col = 0; col < limit; ++col)
        {
            const size_t idx = static_cast<size_t>(col) * limit + row;
            formatted[idx] = FormatComplexCompact(host_matrix[static_cast<size_t>(col) * n + row]);
            widths[static_cast<size_t>(col)] =
                std::max(widths[static_cast<size_t>(col)], formatted[idx].size());
        }
    }

    for (int row = 0; row < limit; ++row)
    {
        for (int col = 0; col < limit; ++col)
        {
            const size_t idx = static_cast<size_t>(col) * limit + row;
            std::cout << std::setw(static_cast<int>(widths[static_cast<size_t>(col)]))
                      << formatted[idx];
            if (col + 1 < limit)
            {
                std::cout << "  ";
            }
        }
        std::cout << std::endl;
    }
}

void PrintGlobalSMatrixTopLeft2x2(const SMatrix& s, int n_smatrix)
{
    PrintTopLeft2x2Compact("S_11(2x2)", s.S_11, n_smatrix);
    PrintTopLeft2x2Compact("S_12(2x2)", s.S_12, n_smatrix);
    PrintTopLeft2x2Compact("S_21(2x2)", s.S_21, n_smatrix);
    PrintTopLeft2x2Compact("S_22(2x2)", s.S_22, n_smatrix);
}

void RequireStatus(Status status, const char* step)
{
    if (status != Status::kSuccess)
    {
        std::cerr << "ERROR: " << step << " failed with status " << static_cast<int>(status)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
} // namespace

__global__ void MagnitudeKernel(const Complex* input, Real* output, int64_t count)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        output[idx] = ComplexAbs(input[idx]);
    }
}

__global__ void IntensityKernel(const Complex* ex,
                                const Complex* ey,
                                const Complex* ez,
                                Real* output,
                                int64_t count)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        const Real exr = ComplexReal(ex[idx]);
        const Real exi = ComplexImag(ex[idx]);
        const Real eyr = ComplexReal(ey[idx]);
        const Real eyi = ComplexImag(ey[idx]);
        const Real ezr = ComplexReal(ez[idx]);
        const Real ezi = ComplexImag(ez[idx]);
        output[idx] = exr * exr + exi * exi + eyr * eyr + eyi * eyi + ezr * ezr + ezi * ezi;
    }
}

struct DeviceBuffer
{
    Complex* ptr = nullptr;
    ~DeviceBuffer() { SAFEFREE(ptr); }
    Complex* get() const { return ptr; }
};

struct DeviceRealBuffer
{
    Real* ptr = nullptr;
    ~DeviceRealBuffer() { SAFEFREE(ptr); }
    Real* get() const { return ptr; }
};

struct ImagLiteral
{
    Real imag;
};

constexpr ImagLiteral operator"" _j(long double imag)
{
    return ImagLiteral{static_cast<Real>(imag)};
}

inline Complex operator+(Real real, ImagLiteral imag) { return make_complex(real, imag.imag); }

void LoadGeometry(std::vector<Real>& geom, int nx, int ny)
{
    geom.assign(static_cast<size_t>(nx) * ny, 0.0);
    for (int x = 0; x < nx; ++x)
    {
        for (int y = ny / 4; y < ny / 2; ++y)
        {
            geom[static_cast<size_t>(x) * ny + y] = 1.0;
        }
    }
    for (int y = 0; y < ny; ++y)
    {
        for (int x = nx / 4; x < nx / 2; ++x)
        {
            geom[static_cast<size_t>(x) * ny + y] = 1.0;
        }
    }
}

void BuildEpsMap(const std::vector<Real>& geom, Complex n, std::vector<Complex>& eps_map)
{
    const size_t count = geom.size();
    eps_map.resize(count);
    const Complex one = make_complex(1.0, 0.0);
    const Complex n2 = ComplexMul(n, n);
    const Complex delta = ComplexSub(n2, one);
    for (size_t i = 0; i < count; ++i)
    {
        const Complex g = make_complex(geom[i], 0.0);
        eps_map[i] = ComplexAdd(one, ComplexMul(delta, g));
    }
}

Real Clamp01(Real v)
{
    if (v < 0.0)
    {
        return 0.0;
    }
    if (v > 1.0)
    {
        return 1.0;
    }
    return v;
}

void JetColor(Real x, unsigned char& r, unsigned char& g, unsigned char& b)
{
    const double t = 4.0 * Clamp01(x);

    const double r_val = Clamp01(std::min(t - 1.5, -t + 4.5));
    const double g_val = Clamp01(std::min(t - 0.5, -t + 3.5));
    const double b_val = Clamp01(std::min(t + 0.5, -t + 2.5));

    const double scale = 255.0;
    r = static_cast<unsigned char>(std::min(scale, r_val * scale));
    g = static_cast<unsigned char>(std::min(scale, g_val * scale));
    b = static_cast<unsigned char>(std::min(scale, b_val * scale));
}

uint32_t Crc32(const unsigned char* data, size_t len)
{
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized)
    {
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
            {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        initialized = true;
    }

    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
    {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

uint32_t Adler32(const unsigned char* data, size_t len)
{
    uint32_t a = 1;
    uint32_t b = 0;
    for (size_t i = 0; i < len; ++i)
    {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void WriteU32BE(std::ofstream& out, uint32_t value)
{
    out.put(static_cast<char>((value >> 24) & 0xFF));
    out.put(static_cast<char>((value >> 16) & 0xFF));
    out.put(static_cast<char>((value >> 8) & 0xFF));
    out.put(static_cast<char>(value & 0xFF));
}

bool WritePng(const char* path, const std::vector<unsigned char>& rgb, int width, int height)
{
    const int row_bytes = width * 3;
    std::vector<unsigned char> raw(static_cast<size_t>(height) * (row_bytes + 1));
    for (int y = 0; y < height; ++y)
    {
        const size_t row_offset = static_cast<size_t>(y) * (row_bytes + 1);
        raw[row_offset] = 0;
        std::copy(rgb.begin() + static_cast<size_t>(y) * row_bytes,
                  rgb.begin() + static_cast<size_t>(y + 1) * row_bytes,
                  raw.begin() + row_offset + 1);
    }

    std::vector<unsigned char> zlib;
    zlib.reserve(raw.size() + raw.size() / 65535 + 16);
    zlib.push_back(0x78);
    zlib.push_back(0x01);

    size_t offset = 0;
    while (offset < raw.size())
    {
        const size_t remaining = raw.size() - offset;
        const uint16_t block_len = static_cast<uint16_t>(std::min<size_t>(remaining, 65535));
        const bool last = (offset + block_len) == raw.size();
        zlib.push_back(last ? 0x01 : 0x00);
        zlib.push_back(static_cast<unsigned char>(block_len & 0xFF));
        zlib.push_back(static_cast<unsigned char>((block_len >> 8) & 0xFF));
        const uint16_t nlen = static_cast<uint16_t>(~block_len);
        zlib.push_back(static_cast<unsigned char>(nlen & 0xFF));
        zlib.push_back(static_cast<unsigned char>((nlen >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.begin() + offset, raw.begin() + offset + block_len);
        offset += block_len;
    }

    const uint32_t adler = Adler32(raw.data(), raw.size());
    zlib.push_back(static_cast<unsigned char>((adler >> 24) & 0xFF));
    zlib.push_back(static_cast<unsigned char>((adler >> 16) & 0xFF));
    zlib.push_back(static_cast<unsigned char>((adler >> 8) & 0xFF));
    zlib.push_back(static_cast<unsigned char>(adler & 0xFF));

    std::ofstream out(path, std::ios::binary);

    const unsigned char signature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    out.write(reinterpret_cast<const char*>(signature), sizeof(signature));

    unsigned char ihdr[13];
    ihdr[0] = static_cast<unsigned char>((width >> 24) & 0xFF);
    ihdr[1] = static_cast<unsigned char>((width >> 16) & 0xFF);
    ihdr[2] = static_cast<unsigned char>((width >> 8) & 0xFF);
    ihdr[3] = static_cast<unsigned char>(width & 0xFF);
    ihdr[4] = static_cast<unsigned char>((height >> 24) & 0xFF);
    ihdr[5] = static_cast<unsigned char>((height >> 16) & 0xFF);
    ihdr[6] = static_cast<unsigned char>((height >> 8) & 0xFF);
    ihdr[7] = static_cast<unsigned char>(height & 0xFF);
    ihdr[8] = 8;
    ihdr[9] = 2;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;

    WriteU32BE(out, 13);
    out.write("IHDR", 4);
    out.write(reinterpret_cast<const char*>(ihdr), sizeof(ihdr));
    unsigned char ihdr_crc_data[17];
    std::copy(reinterpret_cast<const unsigned char*>("IHDR"),
              reinterpret_cast<const unsigned char*>("IHDR") + 4,
              ihdr_crc_data);
    std::copy(ihdr, ihdr + sizeof(ihdr), ihdr_crc_data + 4);
    const uint32_t ihdr_crc = Crc32(ihdr_crc_data, sizeof(ihdr_crc_data));
    WriteU32BE(out, ihdr_crc);

    WriteU32BE(out, static_cast<uint32_t>(zlib.size()));
    out.write("IDAT", 4);
    out.write(reinterpret_cast<const char*>(zlib.data()),
              static_cast<std::streamsize>(zlib.size()));
    std::vector<unsigned char> idat_crc_data(4 + zlib.size());
    std::copy(reinterpret_cast<const unsigned char*>("IDAT"),
              reinterpret_cast<const unsigned char*>("IDAT") + 4,
              idat_crc_data.begin());
    std::copy(zlib.begin(), zlib.end(), idat_crc_data.begin() + 4);
    const uint32_t idat_crc = Crc32(idat_crc_data.data(), idat_crc_data.size());
    WriteU32BE(out, idat_crc);

    WriteU32BE(out, 0);
    out.write("IEND", 4);
    const uint32_t iend_crc = Crc32(reinterpret_cast<const unsigned char*>("IEND"), 4);
    WriteU32BE(out, iend_crc);

    return true;
}

bool WritePngJet(const char* path, const std::vector<Real>& mag, int width, int height)
{
    Real max_val = 0.0;
    for (Real v : mag)
    {
        if (v > max_val)
        {
            max_val = v;
        }
    }
    if (max_val == 0.0)
    {
        max_val = 1.0;
    }

    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < mag.size(); ++i)
    {
        const Real scaled = mag[i] / max_val;
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;
        JetColor(scaled, r, g, b);
        pixels[i * 3 + 0] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }

    return WritePng(path, pixels, width, height);
}

bool WriteMaskPng(const char* path, const std::vector<Real>& mask, int width, int height)
{
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < width; ++i)
    {
        for (size_t j = 0; j < height; ++j)
        {
            const unsigned char c = mask[i * height + j] > 0.5 ? 255 : 0;
            size_t k = j * width + i;
            pixels[k * 3 + 0] = c;
            pixels[k * 3 + 1] = c;
            pixels[k * 3 + 2] = c;
        }
    }
    return WritePng(path, pixels, width, height);
}

bool SaveFieldImage(const char* path,
                    const Complex* field,
                    int width,
                    int height,
                    std::vector<Real>& host_mag,
                    DeviceRealBuffer& device_mag)
{
    const int64_t count = static_cast<int64_t>(width) * height;
    if (!device_mag.get())
    {
        cudaMalloc(reinterpret_cast<void**>(&device_mag.ptr),
                   static_cast<size_t>(count) * sizeof(Real));
    }

    const int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    MagnitudeKernel<<<blocks, CUDA_THREADS>>>(field, device_mag.get(), count);
    host_mag.resize(static_cast<size_t>(count));
    cudaMemcpy(host_mag.data(),
               device_mag.get(),
               static_cast<size_t>(count) * sizeof(Real),
               cudaMemcpyDeviceToHost);

    return WritePngJet(path, host_mag, width, height);
}

bool SaveIntensityImage(const char* path,
                        const Complex* ex,
                        const Complex* ey,
                        const Complex* ez,
                        int width,
                        int height,
                        std::vector<Real>& host_mag,
                        DeviceRealBuffer& device_mag)
{
    const int64_t count = static_cast<int64_t>(width) * height;
    if (!device_mag.get())
    {
        cudaMalloc(reinterpret_cast<void**>(&device_mag.ptr),
                   static_cast<size_t>(count) * sizeof(Real));
    }

    const int blocks = static_cast<int>((count + CUDA_THREADS - 1) / CUDA_THREADS);
    IntensityKernel<<<blocks, CUDA_THREADS>>>(ex, ey, ez, device_mag.get(), count);

    host_mag.resize(static_cast<size_t>(count));
    cudaMemcpy(host_mag.data(),
               device_mag.get(),
               static_cast<size_t>(count) * sizeof(Real),
               cudaMemcpyDeviceToHost);

    return WritePngJet(path, host_mag, width, height);
}

int main()
{
    std::cout << "Precision: "
#if USE_DOUBLE_PRECISION
              << "double" << std::endl;
#else
              << "single" << std::endl;
#endif

    const auto t_pipeline_start = Clock::now();

    const Real period_x = 3000.0;
    const Real period_y = 1500.0;
    const Real wavelength = 365.0;
    const Real freq = 1.0 / wavelength;
    const int order = 13;
    const int order_x = order;
    const int order_y = order;

    RCWA rcwa(freq, order_x, order_y, period_x, period_y);
    rcwa.AddInputLayer(1.0);
    rcwa.SetIncidentAngles(0.0, 0.0, "input");

    const int geom_x = 200;
    const int geom_y = 100;
    std::vector<Real> geometry;
    const auto t_geom_start = Clock::now();
    LoadGeometry(geometry, geom_x, geom_y);
    WriteMaskPng("geometry_mask.png", geometry, geom_x, geom_y);
    const double t_geometry = ElapsedMillis(t_geom_start, Clock::now());

    DeviceBuffer d_material;
    cudaMalloc(reinterpret_cast<void**>(&d_material.ptr),
               static_cast<size_t>(geom_x) * geom_y * sizeof(Complex));

    const Complex n_OX = 1.4745 + 0.0_j;
    const Complex n_TIN = 2.0669 + 1.0563_j;
    const Complex n_TEOS = 1.380 + 0.0_j;
    const Complex n_SIN = 2.1222 + 0.0_j;
    const Complex n_sub = 6.5271 + 2.6672_j;

    const Real thicknesses[] = {100.0, 21.0, 100.0};
    const Complex ns[] = {n_OX, n_TIN, n_TEOS};

    std::vector<Complex> eps_map;
    double t_conv = 0.0;
    double t_scatter = 0.0;

    for (int i = 0; i < 3; ++i)
    {
        BuildEpsMap(geometry, ns[i], eps_map);
        cudaMemcpy(d_material.get(),
                   eps_map.data(),
                   eps_map.size() * sizeof(Complex),
                   cudaMemcpyHostToDevice);

        Complex* eps_conv = nullptr;

        auto t_conv_start = Clock::now();
        RequireStatus(rcwa.MaterialConv(d_material.get(), geom_x, geom_y, &eps_conv),
                      "MaterialConv");
        t_conv += ElapsedMillis(t_conv_start, Clock::now());

        auto t_addlayer_start = Clock::now();
        RequireStatus(rcwa.AddLayer(thicknesses[i], eps_conv), "AddLayer");
        t_scatter += ElapsedMillis(t_addlayer_start, Clock::now());
        SAFEFREE(eps_conv);
    }

    const auto t_hom_start = Clock::now();
    RequireStatus(rcwa.AddLayerHomogeneous(20.0, n_SIN * n_SIN), "AddLayerHomogeneous SIN");
    RequireStatus(rcwa.AddLayerHomogeneous(5.0, n_sub * n_sub), "AddLayerHomogeneous substrate");
    const double t_hom = ElapsedMillis(t_hom_start, Clock::now());

    const auto t_global_start = Clock::now();
    RequireStatus(rcwa.SolveGlobalSMatrix(), "SolveGlobalSMatrix");
    const double t_global = ElapsedMillis(t_global_start, Clock::now());
    PrintGlobalSMatrixTopLeft2x2(rcwa.global_smatrix(), static_cast<int>(2 * rcwa.order_N()));

    const auto t_source_start = Clock::now();
    RequireStatus(rcwa.SourcePlanewave(0.0, 1.0, "forward"), "SourcePlanewave");
    const double t_source = ElapsedMillis(t_source_start, Clock::now());

    const int scatter_x = 256;
    const int scatter_y = 128;
    std::vector<Real> xs(scatter_x);
    std::vector<Real> ys(scatter_y);
    for (int i = 0; i < scatter_x; ++i)
    {
        xs[i] = (period_x / scatter_x) * (i + 0.5);
    }
    for (int j = 0; j < scatter_y; ++j)
    {
        ys[j] = (period_y / scatter_y) * (j + 0.5);
    }

    Complex* Ex = nullptr;
    Complex* Ey = nullptr;
    Complex* Ez = nullptr;
    const auto t_field_start = Clock::now();
    RequireStatus(rcwa.FieldXY(xs.data(), scatter_x, ys.data(), scatter_y, &Ex, &Ey, &Ez),
                  "FieldXY");
    const double t_field = ElapsedMillis(t_field_start, Clock::now());

    std::vector<Real> host_mag;
    DeviceRealBuffer device_mag;
    const auto t_image_start = Clock::now();
    SaveFieldImage("scatter_ex.png", Ex, scatter_x, scatter_y, host_mag, device_mag);
    SaveFieldImage("scatter_ey.png", Ey, scatter_x, scatter_y, host_mag, device_mag);
    SaveFieldImage("scatter_ez.png", Ez, scatter_x, scatter_y, host_mag, device_mag);
    SaveIntensityImage(
        "scatter_intensity.png", Ex, Ey, Ez, scatter_x, scatter_y, host_mag, device_mag);
    const double t_image = ElapsedMillis(t_image_start, Clock::now());

    const double t_pipeline_wall = ElapsedMillis(t_pipeline_start, Clock::now());

    printf("geometry\t%.1f ms\nconv\t%.1f ms\nscatter\t%.1f ms\nhomo\t%.1f ms\n"
           "global\t%.1f ms\nsource\t%.1f ms\nfield\t%.1f ms\nimage\t%.1f ms\n"
           "pipeline_wall\t%.1f ms\n",
           t_geometry,
           t_conv,
           t_scatter,
           t_hom,
           t_global,
           t_source,
           t_field,
           t_image,
           t_pipeline_wall);

    rcwa.FreeDeviceTensor(Ex);
    rcwa.FreeDeviceTensor(Ey);
    rcwa.FreeDeviceTensor(Ez);
    return 0;
}
