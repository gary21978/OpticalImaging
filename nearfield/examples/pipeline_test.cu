#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "rcwa.h"

__global__ void MagnitudeKernel(const Complex* input, Real* output, int64_t count)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        output[idx] = ComplexAbs(input[idx]);
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
    geom.assign(static_cast<size_t>(nx) * ny, kZero);
    for (int x = 0; x < nx; ++x)
    {
        for (int y = 25; y < 50; ++y)
        {
            geom[static_cast<size_t>(x) * ny + y] = kOne;
        }
    }
    for (int x = 50; x < 100; ++x)
    {
        for (int y = 0; y < ny; ++y)
        {
            geom[static_cast<size_t>(x) * ny + y] = kOne;
        }
    }
}

void BuildEpsMap(const std::vector<Real>& geom, Complex n, std::vector<Complex>& eps_map)
{
    const size_t count = geom.size();
    eps_map.resize(count);
    const Complex one = make_complex(kOne, kZero);
    const Complex n2 = ComplexMul(n, n);
    const Complex delta = ComplexSub(n2, one);
    for (size_t i = 0; i < count; ++i)
    {
        const Complex g = make_complex(geom[i], kZero);
        eps_map[i] = ComplexAdd(one, ComplexMul(delta, g));
    }
}

Real Clamp01(Real v)
{
    if (v < kZero)
    {
        return kZero;
    }
    if (v > kOne)
    {
        return kOne;
    }
    return v;
}

void JetColor(Real x, unsigned char& r, unsigned char& g, unsigned char& b)
{
    const Real four = kTwo + kTwo;
    const Real one_point_five = kOne + kHalf;
    const Real two_point_five = kTwo + kHalf;
    const Real three_point_five = kTwo + kOne + kHalf;
    const Real four_point_five = kTwo + kTwo + kHalf;
    const Real t = four * Clamp01(x);

    const Real r_val = Clamp01(std::min(t - one_point_five, -t + four_point_five));
    const Real g_val = Clamp01(std::min(t - kHalf, -t + three_point_five));
    const Real b_val = Clamp01(std::min(t + kHalf, -t + two_point_five));

    const double scale = 255.0;
    r = static_cast<unsigned char>(std::min(scale, static_cast<double>(r_val) * scale));
    g = static_cast<unsigned char>(std::min(scale, static_cast<double>(g_val) * scale));
    b = static_cast<unsigned char>(std::min(scale, static_cast<double>(b_val) * scale));
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
    Real max_val = kZero;
    for (Real v : mag)
    {
        if (v > max_val)
        {
            max_val = v;
        }
    }
    if (max_val == kZero)
    {
        max_val = kOne;
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

    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    MagnitudeKernel<<<blocks, threads>>>(field, device_mag.get(), count);
    host_mag.resize(static_cast<size_t>(count));
    cudaMemcpy(host_mag.data(),
               device_mag.get(),
               static_cast<size_t>(count) * sizeof(Real),
               cudaMemcpyDeviceToHost);

    return WritePngJet(path, host_mag, width, height);
}

int main()
{
    const Real period_x = 3000.0;
    const Real period_y = 1500.0;
    const Real wavelength = 365.0;
    const Real freq = 1.0 / wavelength;
    const int order_x = 5;
    const int order_y = 5;

    RCWA rcwa(freq, order_x, order_y, period_x, period_y);
    rcwa.AddInputLayer(1.0);
    rcwa.SetIncidentAngles(0.0, 0.0, "input");

    const int geom_x = 200;
    const int geom_y = 100;
    std::vector<Real> geometry;
    LoadGeometry(geometry, geom_x, geom_y);

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
    for (int i = 0; i < 3; ++i)
    {
        BuildEpsMap(geometry, ns[i], eps_map);
        cudaMemcpy(d_material.get(),
                   eps_map.data(),
                   eps_map.size() * sizeof(Complex),
                   cudaMemcpyHostToDevice);

        Complex* eps_conv = nullptr;
        rcwa.MaterialConv(d_material.get(), geom_x, geom_y, &eps_conv);
        rcwa.AddLayer(thicknesses[i], eps_conv);
        SAFEFREE(eps_conv);
    }

    rcwa.AddLayerHomogeneous(20.0, n_SIN * n_SIN);
    rcwa.AddLayerHomogeneous(5.0, n_sub * n_sub);

    rcwa.SolveGlobalSMatrix();
    rcwa.SourcePlanewave(0.0, 1.0, "forward");

    const int scatter_x = 256;
    const int scatter_y = 128;
    std::vector<Real> xs(scatter_x);
    std::vector<Real> ys(scatter_y);
    for (int i = 0; i < scatter_x; ++i)
    {
        xs[i] = (period_x / scatter_x) * (i + kHalf);
    }
    for (int j = 0; j < scatter_y; ++j)
    {
        ys[j] = (period_y / scatter_y) * (j + kHalf);
    }

    Complex* Ex = nullptr;
    Complex* Ey = nullptr;
    Complex* Ez = nullptr;
    rcwa.FieldXY(xs.data(), scatter_x, ys.data(), scatter_y, &Ex, &Ey, &Ez);

    std::vector<Real> host_mag;
    DeviceRealBuffer device_mag;
    SaveFieldImage("scatter_ex.png", Ex, scatter_x, scatter_y, host_mag, device_mag);
    SaveFieldImage("scatter_ey.png", Ey, scatter_x, scatter_y, host_mag, device_mag);
    SaveFieldImage("scatter_ez.png", Ez, scatter_x, scatter_y, host_mag, device_mag);

    std::cout << "RCWA pipeline test completed.\n";

    rcwa.FreeDeviceTensor(Ex);
    rcwa.FreeDeviceTensor(Ey);
    rcwa.FreeDeviceTensor(Ez);
    return 0;
}