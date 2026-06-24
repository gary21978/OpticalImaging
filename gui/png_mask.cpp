#include "png_mask.h"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

bool LoadBinaryPngMask(const std::string& path,
                       int target_width,
                       int target_height,
                       std::vector<double>* mask,
                       std::string* error,
                       bool preserve_aspect,
                       int* source_width,
                       int* source_height)
{
    if (!mask || target_width <= 0 || target_height <= 0)
        return false;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
    {
        if (error)
            *error = "Cannot open PNG: " + path;
        return false;
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info)
    {
        std::fclose(fp);
        if (error)
            *error = "libpng initialization failed.";
        return false;
    }
    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        if (error)
            *error = "Failed to decode PNG: " + path;
        return false;
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    const int width = static_cast<int>(png_get_image_width(png, info));
    const int height = static_cast<int>(png_get_image_height(png, info));
    if (source_width)
        *source_width = width;
    if (source_height)
        *source_height = height;
    const png_byte color_type = png_get_color_type(png, info);
    const png_byte bit_depth = png_get_bit_depth(png, info);
    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA)
        png_set_rgb_to_gray_fixed(png, 1, -1, -1);
    if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_strip_alpha(png);
    if (bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    png_read_update_info(png, info);

    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height);
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    for (int y = 0; y < height; ++y)
        rows[static_cast<size_t>(y)] = pixels.data() + static_cast<size_t>(y) * width;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    int output_width = target_width;
    int output_height = target_height;
    int offset_x = 0;
    int offset_y = 0;
    if (preserve_aspect)
    {
        const double scale = std::min(static_cast<double>(target_width) / width,
                                      static_cast<double>(target_height) / height);
        output_width = std::max(1, static_cast<int>(std::lround(width * scale)));
        output_height = std::max(1, static_cast<int>(std::lround(height * scale)));
        offset_x = (target_width - output_width) / 2;
        offset_y = (target_height - output_height) / 2;
    }

    mask->assign(static_cast<size_t>(target_width) * target_height, 0.0);
    for (int y = 0; y < output_height; ++y)
    {
        const int sy = std::clamp(y * height / output_height, 0, height - 1);
        for (int x = 0; x < output_width; ++x)
        {
            const int sx = std::clamp(x * width / output_width, 0, width - 1);
            const unsigned char value = pixels[static_cast<size_t>(sy) * width + sx];
            (*mask)[static_cast<size_t>(x + offset_x) * target_height + y + offset_y] =
                value >= 128 ? 1.0 : 0.0;
        }
    }
    if (error)
        error->clear();
    return true;
}
