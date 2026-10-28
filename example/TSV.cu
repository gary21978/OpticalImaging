#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "abbe_imager.h"

namespace tsv_image {
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

    if (cudaMemcpy(host_ex.data(), ex, count * sizeof(Complex), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_ey.data(), ey, count * sizeof(Complex), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_ez.data(), ez, count * sizeof(Complex), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("near-field image download failed");

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

} // namespace tsv_image

#include <filesystem>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace {
struct DeviceBuffer {
    Complex* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};
Status StatusFromCudaLocal(cudaError_t s) { return s == cudaSuccess ? Status::kSuccess : Status::kCudaError; }
using namespace tsv_image;
} // namespace

void TSVRequire(Status s, const char* step)
{
    if (s != Status::kSuccess)
        throw std::runtime_error(std::string(step) + ": " + StatusToString(s));
}
int main(int argc, char** argv)
try
{
    if (argc > 6)
        throw std::runtime_error(
            "usage: TSV [order] [output_dir] [depth_um] [n_real] [n_imag]");
    const int order =
        argc > 1 ? std::stoi(argv[1]) : 30; // precision is selected by USE_DOUBLE_PRECISION
    const std::filesystem::path out = argc > 2 ? argv[2] : "doc/results/tsv_si_M30";
    if (order < 1 || order > 255) throw std::runtime_error("order must be 1..255");
    std::filesystem::create_directories(out);
    std::cout << std::unitbuf;
    constexpr int g = 1024, w = 301, h = 301;
    constexpr Real period = static_cast<Real>(12.3), lambda = static_cast<Real>(0.53);
    const Real depth = argc > 3 ? static_cast<Real>(std::stod(argv[3])) : 30.0f;
    if (!std::isfinite(static_cast<double>(depth)) || depth <= 0)
        throw std::runtime_error("depth_um must be finite and positive");
    const Real n_real = argc > 4 ? static_cast<Real>(std::stod(argv[4])) : static_cast<Real>(4.15);
    const Real n_imag =
        argc > 5 ? static_cast<Real>(std::stod(argv[5])) : (argc > 4 ? 0.0f : static_cast<Real>(0.033));
    if (!std::isfinite(static_cast<double>(n_real)) ||
        !std::isfinite(static_cast<double>(n_imag)) || n_real <= 0 || n_imag < 0)
        throw std::runtime_error("require finite n_real>0 and n_imag>=0");
    const Complex n = make_complex(n_real, n_imag), eps_material = n * n;
    const bool high_order =
        !std::getenv("TSV_HIGH_ORDER") || std::string(std::getenv("TSV_HIGH_ORDER")) != "0";
    std::ofstream metadata(out / "case.txt");
    metadata << std::setprecision(12) << "TSV: M=" << order << "; "
             << (sizeof(Real) == 4 ? "float32" : "float64") << "; Lalanne; FREE"
             << (high_order ? 18 : 12) << "; geometry=1024x1024\n"
             << "n=" << n_real << "+" << n_imag << "i; epsilon=" << ComplexReal(eps_material) << "+"
             << ComplexImag(eps_material) << "i\n"
             << "Air hole diameter=3 um; depth=" << depth
             << " um; period=12.3x12.3 um; semi-infinite same-material substrate\n"
             << "lambda=0.53 um; normal incidence; Y polarization; illumination NA=0.1; imaging "
                "NA=0.9; magnification=100; zero aberration\n"
             << "Coordinates in images.csv are object-space um; intensity is raw, not peak "
                "normalized.\n";
    metadata.close();
    RCWA r(1 / lambda, order, order, period, period);
    r.SetExpmMethod('F');
    r.SetHighOrder(high_order);
    TSVRequire(r.AddInputLayer(1), "input");
    TSVRequire(r.AddOutputLayer(eps_material), "substrate");
    TSVRequire(r.SetIncidentAngles(0, 0, "input"), "incident angles");
    std::vector<Complex> eps(g * g, eps_material);
    std::vector<Real> mask(g * g, 0);
    for (int x = 0; x < g; ++x)
        for (int y = 0; y < g; ++y)
        {
            Real xx = (x + .5f) * period / g - period / 2, yy = (y + .5f) * period / g - period / 2;
            if (xx * xx + yy * yy < 2.25f)
            {
                eps[x * g + y] = make_complex(1, 0);
                mask[x * g + y] = 1;
            }
        }
    DeviceBuffer device;
    TSVRequire(StatusFromCudaLocal(
                   cudaMalloc(reinterpret_cast<void**>(&device.ptr), eps.size() * sizeof(Complex))),
               "geometry allocation");
    TSVRequire(StatusFromCudaLocal(cudaMemcpy(
                   device.ptr, eps.data(), eps.size() * sizeof(Complex), cudaMemcpyHostToDevice)),
               "geometry upload");
    TSVRequire(r.AddLayer(depth, device.ptr, g, g, LayerFactorization::kLalanne), "TSV layer");
    TSVRequire(r.SolveGlobalSMatrix(), "global solve");
    // The Abbe path packs only illumination-pupil columns from the possibly
    // out-of-core final S matrix; it does not materialize all four S blocks.
    std::vector<Real> xs(w), ys(h);
    for (int i = 0; i < w; ++i) xs[i] = -period / 2 + (i + .5f) * period / w;
    for (int i = 0; i < h; ++i) ys[i] = -period / 2 + (i + .5f) * period / h;
    AbbeSourceConfig src;
    src.sigma_in = 0;
    src.sigma_out = .1;
    src.polarization_mode = AbbePolarizationMode::kY;
    AbbeProjectorConfig proj;
    proj.wavelength = lambda;
    proj.magnification = 100;
    proj.sigma_in = 0;
    proj.sigma_out = .9;
    proj.focus_range = {0};
    proj.aberration_zernike.assign(37, 0);
    AbbeNumericsConfig nums;
    AbbeImageResult far;
    TSVRequire(CalculateAbbeImageFromRCWA(r, xs.data(), w, ys.data(), h, src, proj, nums, &far),
               "Abbe image");
    TSVRequire(r.SourcePlanewave(0, 1, "forward"), "Y source");
    Complex *ex = nullptr, *ey = nullptr, *ez = nullptr;
    TSVRequire(r.FieldXY(xs.data(), w, ys.data(), h, &ex, &ey, &ez), "near field");
    TSVRequire(StatusFromCudaLocal(cudaDeviceSynchronize()), "near field sync");
    auto near = ComputeIntensityImage(ex, ey, ez, w, h);
    r.FreeDeviceTensor(ex);
    r.FreeDeviceTensor(ey);
    r.FreeDeviceTensor(ez);
    std::ofstream images(out / "images.csv");
    images << "x_um,y_um,near_intensity,abbe_intensity\n" << std::setprecision(12);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            size_t j = y * w + x;
            if (!std::isfinite(static_cast<double>(near[j])) ||
                !std::isfinite(static_cast<double>(far.intensity[j])))
                throw std::runtime_error("nonfinite image");
            images << xs[x] << ',' << ys[y] << ',' << near[j] << ',' << far.intensity[j] << '\n';
        }
    if (!WriteMaskPngGray((out / "mask.png").string(), mask, g, g) ||
        !WritePngGray((out / "nearfield.png").string(), near, w, h) ||
        !WritePngGray((out / "farfield.png").string(), far.intensity, w, h))
        throw std::runtime_error("PNG output");
    std::cout << "Images written to " << out << '\n';
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "TSV failed: " << e.what() << '\n';
    return 1;
}
