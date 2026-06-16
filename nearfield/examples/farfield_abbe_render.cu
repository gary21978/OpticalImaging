#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "abbe_imager.h"

namespace
{
using Clock = std::chrono::steady_clock;

struct DeviceBuffer
{
    Complex* ptr = nullptr;
    ~DeviceBuffer()
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }
};

struct DefaultRenderConfig
{
    Real period_x = 3000.0;
    Real period_y = 1500.0;
    Real wavelength = 365.0;
    int order_x = 12;
    int order_y = 12;
    Real sigma_out = 0.9;
    Real sigma_in = 0.15;
    int geom_x = 200;
    int geom_y = 100;
    int image_w = 257;
    int image_h = 129;
    const char* output_dir = "output";
};

Status StatusFromCudaLocal(cudaError_t status)
{
    return status == cudaSuccess ? Status::kSuccess : Status::kCudaError;
}

void Require(Status status, const char* step)
{
    if (status != Status::kSuccess)
    {
        std::cerr << step << " failed: " << StatusToString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

double ElapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
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

void GrayColor(Real x, unsigned char& r, unsigned char& g, unsigned char& b)
{
    const unsigned char gray =
        static_cast<unsigned char>(std::min(255.0, static_cast<double>(Clamp01(x)) * 255.0));
    r = gray;
    g = gray;
    b = gray;
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

bool WritePng(const std::string& path, const std::vector<unsigned char>& rgb, int width, int height)
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
    if (!out)
    {
        return false;
    }

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
    WriteU32BE(out, Crc32(ihdr_crc_data, sizeof(ihdr_crc_data)));

    WriteU32BE(out, static_cast<uint32_t>(zlib.size()));
    out.write("IDAT", 4);
    out.write(reinterpret_cast<const char*>(zlib.data()), static_cast<std::streamsize>(zlib.size()));
    std::vector<unsigned char> idat_crc_data(4 + zlib.size());
    std::copy(reinterpret_cast<const unsigned char*>("IDAT"),
              reinterpret_cast<const unsigned char*>("IDAT") + 4,
              idat_crc_data.begin());
    std::copy(zlib.begin(), zlib.end(), idat_crc_data.begin() + 4);
    WriteU32BE(out, Crc32(idat_crc_data.data(), idat_crc_data.size()));

    WriteU32BE(out, 0);
    out.write("IEND", 4);
    WriteU32BE(out, Crc32(reinterpret_cast<const unsigned char*>("IEND"), 4));
    return true;
}

bool WritePngGray(const std::string& path, const std::vector<Real>& image, int width, int height)
{
    Real max_val = 0.0;
    for (Real v : image)
    {
        max_val = std::max(max_val, v);
    }
    if (max_val == 0.0)
    {
        max_val = 1.0;
    }

    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < image.size(); ++i)
    {
        unsigned char r = 0;
        unsigned char g = 0;
        unsigned char b = 0;
        GrayColor(image[i] / max_val, r, g, b);
        pixels[i * 3 + 0] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }
    return WritePng(path, pixels, width, height);
}

bool WriteMaskPngGray(const std::string& path, const std::vector<Real>& mask, int width, int height)
{
    if (static_cast<int>(mask.size()) != width * height)
    {
        return false;
    }

    std::vector<Real> raster(static_cast<size_t>(width) * height, 0.0);
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            const size_t src = static_cast<size_t>(x) * height + y;
            const size_t dst = static_cast<size_t>(y) * width + x;
            raster[dst] = mask[src] > 0.5 ? 1.0 : 0.0;
        }
    }
    return WritePngGray(path, raster, width, height);
}

std::vector<Real> BuildAxis(int count, Real length)
{
    std::vector<Real> axis(static_cast<size_t>(count), 0.0);
    const Real step = length / static_cast<Real>(count);
    for (int i = 0; i < count; ++i)
    {
        axis[static_cast<size_t>(i)] = step * static_cast<Real>(i + 0.5);
    }
    return axis;
}

std::vector<Real> ComputeIntensityImage(const Complex* ex,
                                        const Complex* ey,
                                        const Complex* ez,
                                        int width,
                                        int height)
{
    const size_t count = static_cast<size_t>(width) * height;
    std::vector<Complex> host_ex(count);
    std::vector<Complex> host_ey(count);
    std::vector<Complex> host_ez(count);
    std::vector<Real> intensity(count, 0.0);

    cudaMemcpy(host_ex.data(), ex, count * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ey.data(), ey, count * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ez.data(), ez, count * sizeof(Complex), cudaMemcpyDeviceToHost);

    auto norm2 = [](Complex v) {
        const Real re = ComplexReal(v);
        const Real im = ComplexImag(v);
        return re * re + im * im;
    };

    for (size_t i = 0; i < count; ++i)
    {
        intensity[i] = norm2(host_ex[i]) + norm2(host_ey[i]) + norm2(host_ez[i]);
    }
    return intensity;
}

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
    const Complex one = make_complex(1.0, 0.0);
    const Complex n2 = ComplexMul(n, n);
    const Complex delta = ComplexSub(n2, one);
    eps_map.resize(geom.size());
    for (size_t i = 0; i < geom.size(); ++i)
    {
        eps_map[i] = ComplexAdd(one, ComplexMul(make_complex(geom[i], 0.0), delta));
    }
}

} // namespace

int main(int argc, char** argv)
{
    const auto pipeline_start = Clock::now();
    DefaultRenderConfig config;

    if (argc > 2)
    {
        std::cerr << "usage: " << argv[0] << " [output_dir]" << std::endl;
        return EXIT_FAILURE;
    }
    const std::filesystem::path out_dir =
        std::filesystem::current_path() / (argc == 2 ? argv[1] : config.output_dir);
    const Real freq = 1.0 / config.wavelength;

    const auto rcwa_setup_start = Clock::now();
    RCWA rcwa(freq, config.order_x, config.order_y, config.period_x, config.period_y);
    Require(rcwa.AddInputLayer(1.0), "AddInputLayer");
    Require(rcwa.AddOutputLayer(
                ComplexMul(make_complex(6.5271, 2.6672), make_complex(6.5271, 2.6672))),
            "AddOutputLayer");
    Require(rcwa.SetIncidentAngles(0.0, 0.0, "input"), "SetIncidentAngles");
    const auto rcwa_setup_end = Clock::now();

    const auto geometry_start = Clock::now();
    std::vector<Real> geometry;
    LoadGeometry(geometry, config.geom_x, config.geom_y);

    DeviceBuffer d_material;
    cudaMalloc(
        reinterpret_cast<void**>(&d_material.ptr),
        static_cast<size_t>(config.geom_x) * config.geom_y * sizeof(Complex));

    const Complex n_OX = make_complex(1.4745, 0.0);
    const Complex n_TIN = make_complex(2.0669, 1.0563);
    const Complex n_TEOS = make_complex(1.380, 0.0);
    const Real thicknesses[] = {100.0, 21.0, 100.0};
    const Complex ns[] = {n_OX, n_TIN, n_TEOS};
    std::vector<Complex> eps_map;
    for (int i = 0; i < 3; ++i)
    {
        BuildEpsMap(geometry, ns[i], eps_map);
        cudaMemcpy(d_material.ptr,
                   eps_map.data(),
                   eps_map.size() * sizeof(Complex),
                   cudaMemcpyHostToDevice);
        Complex* eps_conv = nullptr;
        Require(rcwa.MaterialConv(d_material.ptr, config.geom_x, config.geom_y, &eps_conv), "MaterialConv");
        Require(rcwa.AddLayer(thicknesses[i], eps_conv), "AddLayer");
        rcwa.FreeDeviceTensor(eps_conv);
    }
    Require(rcwa.AddLayerHomogeneous(
                20.0, ComplexMul(make_complex(2.1222, 0.0), make_complex(2.1222, 0.0))),
            "AddLayerHomogeneous");
    Require(StatusFromCudaLocal(cudaDeviceSynchronize()), "GeometryAndLayerSync");
    const auto geometry_end = Clock::now();

    const auto smatrix_start = Clock::now();
    Require(rcwa.SolveGlobalSMatrix(), "SolveGlobalSMatrix");
    Require(StatusFromCudaLocal(cudaDeviceSynchronize()), "SolveGlobalSMatrixSync");
    const auto smatrix_end = Clock::now();

    const auto imaging_grid_start = Clock::now();
    const std::vector<Real> xs = BuildAxis(config.image_w, config.period_x);
    const std::vector<Real> ys = BuildAxis(config.image_h, config.period_y);
    const auto imaging_grid_end = Clock::now();

    AbbeSourceConfig source;
    source.shape = AbbeApertureShape::kAnnular;
    source.sigma_out = config.sigma_out;
    source.sigma_in = config.sigma_in;
    source.polarization_mode = AbbePolarizationMode::kX;

    AbbeProjectorConfig projector;
    projector.wavelength = config.wavelength;
    projector.shape = AbbeApertureShape::kAnnular;
    projector.sigma_out = config.sigma_out;
    projector.sigma_in = config.sigma_in;
    projector.focus_range = {0.0};

    AbbeNumericsConfig numerics;
    numerics.object_z_positions = {0.0};
    AbbeImager imager(source, projector, numerics);
    const auto abbe_start = Clock::now();
    Require(imager.ComputeFromRCWA(
                rcwa, xs.data(), config.image_w, ys.data(), config.image_h),
            "ComputeFromRCWA");
    Require(StatusFromCudaLocal(cudaDeviceSynchronize()), "ComputeFromRCWASync");
    const auto abbe_end = Clock::now();

    const auto nearfield_start = Clock::now();
    Require(rcwa.SourcePlanewave(1.0, 0.0, "forward"), "SourcePlanewave");
    Complex* ex = nullptr;
    Complex* ey = nullptr;
    Complex* ez = nullptr;
    Require(rcwa.FieldXY(xs.data(), config.image_w, ys.data(), config.image_h, &ex, &ey, &ez), "FieldXY");
    Require(StatusFromCudaLocal(cudaDeviceSynchronize()), "FieldXYSync");
    const std::vector<Real> nearfield_intensity = ComputeIntensityImage(ex, ey, ez, config.image_w, config.image_h);
    rcwa.FreeDeviceTensor(ex);
    rcwa.FreeDeviceTensor(ey);
    rcwa.FreeDeviceTensor(ez);
    const auto nearfield_end = Clock::now();

    const auto output_start = Clock::now();
    const std::vector<Real> intensity = imager.GetFocusSlice(0);
    std::filesystem::create_directories(out_dir);
    const std::filesystem::path mask_png = out_dir / "mask.png";
    const std::filesystem::path near_png = out_dir / "nearfield.png";
    const std::filesystem::path far_png = out_dir / "farfield.png";
    if (!WriteMaskPngGray(mask_png.string(), geometry, config.geom_x, config.geom_y))
    {
        std::cerr << "failed to write " << mask_png << std::endl;
        return EXIT_FAILURE;
    }
    if (!WritePngGray(near_png.string(), nearfield_intensity, config.image_w, config.image_h))
    {
        std::cerr << "failed to write " << near_png << std::endl;
        return EXIT_FAILURE;
    }
    if (!WritePngGray(far_png.string(), intensity, config.image_w, config.image_h))
    {
        std::cerr << "failed to write " << far_png << std::endl;
        return EXIT_FAILURE;
    }
    const auto output_end = Clock::now();

    const double rcwa_setup_ms = ElapsedMs(rcwa_setup_start, rcwa_setup_end);
    const double geometry_ms = ElapsedMs(geometry_start, geometry_end);
    const double smatrix_ms = ElapsedMs(smatrix_start, smatrix_end);
    const double imaging_grid_ms = ElapsedMs(imaging_grid_start, imaging_grid_end);
    const double abbe_ms = ElapsedMs(abbe_start, abbe_end);
    const double nearfield_ms = ElapsedMs(nearfield_start, nearfield_end);
    const double output_ms = ElapsedMs(output_start, output_end);
    const double elapsed_ms = ElapsedMs(pipeline_start, output_end);

    std::cout << "Saved mask to: " << mask_png << "\n";
    std::cout << "Saved near-field intensity to: " << near_png << "\n";
    std::cout << "Saved far-field intensity to: " << far_png << "\n";
    std::cout << "Image size: " << config.image_w << "x" << config.image_h << "\n";
    std::cout << "Period: (" << config.period_x << ", " << config.period_y << ") nm\n";
    std::cout << "Wavelength: " << config.wavelength << " nm\n";
    std::cout << "RCWA orders: (" << config.order_x << ", " << config.order_y << ")\n";
    std::cout << "Source/projector sigma: outer=" << config.sigma_out << ", inner=" << config.sigma_in << "\n";
    std::cout << "Timing (ms):\n";
    std::cout << "  RCWA setup: " << rcwa_setup_ms << "\n";
    std::cout << "  Geometry/material/layer build: " << geometry_ms << "\n";
    std::cout << "  SolveGlobalSMatrix: " << smatrix_ms << "\n";
    std::cout << "  Image grid setup: " << imaging_grid_ms << "\n";
    std::cout << "  Abbe imaging: " << abbe_ms << "\n";
    std::cout << "  Near-field solve: " << nearfield_ms << "\n";
    std::cout << "  PNG write: " << output_ms << "\n";
    std::cout << "  Total pipeline: " << elapsed_ms << "\n";

    return EXIT_SUCCESS;
}
