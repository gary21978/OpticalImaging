#include "simulation_backend.h"
#include "simulation_config.h"
#include "material_database.h"
#include "png_mask.h"

#include <GLFW/glfw3.h>
#include <png.h>
#include <ft2build.h>
#include FT_FREETYPE_H

#include <array>
#include <algorithm>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr int kGeometryTab = 0;
constexpr int kBasicTab = 1;
constexpr int kSourceTab = 2;
constexpr int kPolarizationTab = 3;
constexpr int kZernikeTab = 4;
constexpr int kSolverTab = 5;

struct Rect
{
    float x, y, w, h;
    bool Contains(double px, double py) const
    {
        return px >= x && px <= x + w && py >= y && py <= y + h;
    }
};

struct Color
{
    float r, g, b, a;
};

Color MaterialColor(const std::string& material, const std::array<float, 3>& fallback)
{
    static const Color palette[] = {
        {0.14f,0.62f,0.91f,1}, {0.93f,0.39f,0.27f,1},
        {0.24f,0.72f,0.48f,1}, {0.62f,0.43f,0.91f,1},
        {0.95f,0.66f,0.20f,1}, {0.16f,0.76f,0.78f,1},
        {0.88f,0.35f,0.68f,1}, {0.48f,0.58f,0.72f,1},
        {0.56f,0.72f,0.20f,1}, {0.85f,0.49f,0.16f,1},
        {0.22f,0.48f,0.82f,1}, {0.68f,0.32f,0.35f,1},
    };
    if (material == "custom")
        return {fallback[0], fallback[1], fallback[2], 1.0f};
    const auto& records = SharedMaterialDatabase().materials();
    for (size_t i = 0; i < records.size(); ++i)
        if (records[i].name == material)
            return palette[i % std::size(palette)];
    return {fallback[0], fallback[1], fallback[2], 1.0f};
}

struct NumericField
{
    std::string label;
    double* real_value = nullptr;
    int* int_value = nullptr;
    Rect rect{};
    bool enabled = true;
};

struct TextField
{
    std::string label;
    std::string* value = nullptr;
    Rect rect{};
};

struct Vec3
{
    float x = 0, y = 0, z = 0;
};

struct Quat
{
    float w = 1, x = 0, y = 0, z = 0;
};

struct GlyphInfo
{
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
    int width = 0, height = 0;
    int bearing_x = 0, bearing_y = 0;
    int advance = 0;
};

struct FontAtlas
{
    GLuint texture = 0;
    int pixel_size = 24;
    int atlas_width = 1024;
    int atlas_height = 1024;
    std::array<GlyphInfo, 128> glyphs{};

    bool Initialize(const char* path)
    {
        FT_Library library = nullptr;
        FT_Face face = nullptr;
        if (FT_Init_FreeType(&library) != 0 || FT_New_Face(library, path, 0, &face) != 0)
        {
            if (library)
                FT_Done_FreeType(library);
            return false;
        }
        FT_Set_Pixel_Sizes(face, 0, pixel_size);
        std::vector<unsigned char> atlas(static_cast<size_t>(atlas_width) * atlas_height, 0);
        int pen_x = 2;
        int pen_y = 2;
        int row_height = 0;
        for (unsigned int c = 32; c < 127; ++c)
        {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER | FT_LOAD_TARGET_LIGHT) != 0)
                continue;
            const FT_GlyphSlot slot = face->glyph;
            if (pen_x + static_cast<int>(slot->bitmap.width) + 2 >= atlas_width)
            {
                pen_x = 2;
                pen_y += row_height + 2;
                row_height = 0;
            }
            if (pen_y + static_cast<int>(slot->bitmap.rows) + 2 >= atlas_height)
                break;
            for (unsigned int row = 0; row < slot->bitmap.rows; ++row)
                for (unsigned int col = 0; col < slot->bitmap.width; ++col)
                    atlas[static_cast<size_t>(pen_y + row) * atlas_width + pen_x + col] =
                        slot->bitmap.buffer[static_cast<size_t>(row) * slot->bitmap.pitch + col];
            GlyphInfo& glyph = glyphs[c];
            glyph.u0 = static_cast<float>(pen_x) / atlas_width;
            glyph.v0 = static_cast<float>(pen_y) / atlas_height;
            glyph.u1 = static_cast<float>(pen_x + slot->bitmap.width) / atlas_width;
            glyph.v1 = static_cast<float>(pen_y + slot->bitmap.rows) / atlas_height;
            glyph.width = static_cast<int>(slot->bitmap.width);
            glyph.height = static_cast<int>(slot->bitmap.rows);
            glyph.bearing_x = slot->bitmap_left;
            glyph.bearing_y = slot->bitmap_top;
            glyph.advance = static_cast<int>(slot->advance.x >> 6);
            pen_x += glyph.width + 2;
            row_height = std::max(row_height, glyph.height);
        }
        FT_Done_Face(face);
        FT_Done_FreeType(library);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, atlas_width, atlas_height, 0,
                     GL_ALPHA, GL_UNSIGNED_BYTE, atlas.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        return true;
    }
};

struct App
{
    GLFWwindow* window = nullptr;
    SimulationConfig config = DefaultSimulationConfig();
    std::string json_path;
    std::string status = "READY - EDIT PARAMETERS OR LOAD JSON";
    std::vector<NumericField> fields;
    std::vector<TextField> text_fields;
    int active_field = -1;
    int active_text_field = -1;
    int active_geom_thickness_row = -1;
    bool period_xy_linked = false;
    bool order_xy_linked = false;
    bool material_menu_open = false;
    int material_menu_row = -1;
    int material_menu_scroll = 0;
    std::string edit_buffer;
    bool replace_numeric_input = false;
    bool mouse_pressed = false;
    bool dragging_view = false;
    double mouse_x = 0.0;
    double mouse_y = 0.0;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    Quat orientation{};
    Quat spin{};
    float camera_zoom = 1.0f;
    Rect structure_panel{};
    std::chrono::steady_clock::time_point last_frame = std::chrono::steady_clock::now();
    int active_tab = 0;
    int zernike_page = 0;
    int selected_focus = 0;
    int selected_layer = 0;
    pid_t worker_pid = -1;
    std::string executable_dir;
    std::string worker_config_path;
    std::string worker_result_path;
    bool running = false;
    SimulationResult result;
    GLuint farfield_texture = 0;
    GLuint nearfield_texture = 0;
    int texture_width = 0;
    int texture_height = 0;
    std::string render_gpu;
    std::string compute_gpu;
};

App* g_app = nullptr;
FontAtlas g_font;

void Quad(Rect r, Color c)
{
    glColor4f(c.r, c.g, c.b, c.a);
    glBegin(GL_QUADS);
    glVertex2f(r.x, r.y);
    glVertex2f(r.x + r.w, r.y);
    glVertex2f(r.x + r.w, r.y + r.h);
    glVertex2f(r.x, r.y + r.h);
    glEnd();
}

void GradientQuad(Rect r, Color top, Color bottom)
{
    glBegin(GL_QUADS);
    glColor4f(top.r, top.g, top.b, top.a);
    glVertex2f(r.x, r.y);
    glVertex2f(r.x + r.w, r.y);
    glColor4f(bottom.r, bottom.g, bottom.b, bottom.a);
    glVertex2f(r.x + r.w, r.y + r.h);
    glVertex2f(r.x, r.y + r.h);
    glEnd();
}

void RoundRect(Rect r, float radius, Color c)
{
    (void)radius;
    Quad(r, c);
}

void RoundOutline(Rect r, float radius, Color c, float width = 1.0f)
{
    (void)radius;
    glColor4f(c.r, c.g, c.b, c.a);
    glLineWidth(width);
    glBegin(GL_LINE_LOOP);
    glVertex2f(r.x, r.y);
    glVertex2f(r.x + r.w, r.y);
    glVertex2f(r.x + r.w, r.y + r.h);
    glVertex2f(r.x, r.y + r.h);
    glEnd();
}

void Shadow(Rect r, float radius, Color color)
{
    (void)r;
    (void)radius;
    (void)color;
}

void GlowLine(float x0, float y0, float x1, float y1, Color color)
{
    glLineWidth(1.0f);
    glColor4f(color.r,color.g,color.b,color.a);
    glBegin(GL_LINES);
    glVertex2f(x0,y0); glVertex2f(x1,y1);
    glEnd();
}

void Outline(Rect r, Color c, float width = 1.0f)
{
    glColor4f(c.r, c.g, c.b, c.a);
    glLineWidth(width);
    glBegin(GL_LINE_LOOP);
    glVertex2f(r.x, r.y);
    glVertex2f(r.x + r.w, r.y);
    glVertex2f(r.x + r.w, r.y + r.h);
    glVertex2f(r.x, r.y + r.h);
    glEnd();
}

float FontScale(float legacy_scale)
{
    return legacy_scale * 9.0f / g_font.pixel_size;
}

float TextWidth(const std::string& text, float legacy_scale)
{
    const float scale = FontScale(legacy_scale);
    float width = 0.0f;
    for (unsigned char c : text)
        width += g_font.glyphs[c < 128 ? c : '?'].advance * scale;
    return width;
}

void Text(float x, float y, const std::string& text, float legacy_scale, Color color)
{
    if (!g_font.texture)
        return;
    const float scale = FontScale(legacy_scale);
    const float baseline = std::round(y + g_font.pixel_size * 0.82f * scale);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_font.texture);
    glColor4f(color.r, color.g, color.b, color.a);
    glBegin(GL_QUADS);
    float cursor = x;
    for (unsigned char c : text)
    {
        const GlyphInfo& glyph = g_font.glyphs[c < 128 ? c : '?'];
        const float x0 = std::round(cursor + glyph.bearing_x * scale);
        const float y0 = std::round(baseline - glyph.bearing_y * scale);
        const float x1 = std::round(x0 + glyph.width * scale);
        const float y1 = std::round(y0 + glyph.height * scale);
        glTexCoord2f(glyph.u0, glyph.v0); glVertex2f(x0, y0);
        glTexCoord2f(glyph.u1, glyph.v0); glVertex2f(x1, y0);
        glTexCoord2f(glyph.u1, glyph.v1); glVertex2f(x1, y1);
        glTexCoord2f(glyph.u0, glyph.v1); glVertex2f(x0, y1);
        cursor = std::round(cursor + glyph.advance * scale);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

Vec3 Cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

float Dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Normalize(Vec3 v)
{
    const float length = std::sqrt(std::max(Dot(v, v), 1.0e-12f));
    return {v.x / length, v.y / length, v.z / length};
}

Quat Normalize(Quat q)
{
    const float length = std::sqrt(std::max(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z, 1.0e-12f));
    return {q.w/length, q.x/length, q.y/length, q.z/length};
}

Quat Multiply(Quat a, Quat b)
{
    return Normalize({
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    });
}

Quat AxisAngle(Vec3 axis, float angle)
{
    axis = Normalize(axis);
    const float half = angle * 0.5f;
    const float s = std::sin(half);
    return Normalize({std::cos(half), axis.x*s, axis.y*s, axis.z*s});
}

Quat SlerpIdentity(Quat q, float amount)
{
    q = Normalize(q);
    if (q.w < 0.0f)
        q = {-q.w,-q.x,-q.y,-q.z};
    const float angle = 2.0f * std::acos(std::clamp(q.w, -1.0f, 1.0f));
    const float s = std::sqrt(std::max(1.0f - q.w*q.w, 0.0f));
    if (s < 1.0e-5f || angle < 1.0e-5f)
        return {};
    return AxisAngle({q.x/s,q.y/s,q.z/s}, angle * amount);
}

void QuatMatrix(Quat q, float* m)
{
    q = Normalize(q);
    const float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    const float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    const float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    const float out[16] = {
        1-2*(yy+zz), 2*(xy+wz), 2*(xz-wy), 0,
        2*(xy-wz), 1-2*(xx+zz), 2*(yz+wx), 0,
        2*(xz+wy), 2*(yz-wx), 1-2*(xx+yy), 0,
        0,0,0,1
    };
    std::copy(out, out + 16, m);
}

Vec3 TrackballPoint(const Rect& panel, double px, double py)
{
    const float radius = 0.92f * std::min(panel.w, panel.h) * 0.5f;
    float x = static_cast<float>(px - (panel.x + panel.w * 0.5f)) / radius;
    float y = static_cast<float>((panel.y + panel.h * 0.5f) - py) / radius;
    const float d2 = x*x + y*y;
    const float z = d2 <= 1.0f ? std::sqrt(1.0f - d2) : 0.0f;
    if (d2 > 1.0f)
    {
        const float inv = 1.0f / std::sqrt(d2);
        x *= inv;
        y *= inv;
    }
    return Normalize(Vec3{x,y,z});
}

Quat TrackballDelta(const Rect& panel, double x0, double y0, double x1, double y1)
{
    const Vec3 a = TrackballPoint(panel, x0, y0);
    const Vec3 b = TrackballPoint(panel, x1, y1);
    const Vec3 axis = Cross(a, b);
    const float axis_length = std::sqrt(Dot(axis, axis));
    if (axis_length < 1.0e-5f)
        return {};
    return AxisAngle(axis, std::acos(std::clamp(Dot(a,b), -1.0f, 1.0f)));
}

std::string Format(double value, int precision = 3)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string OpenPngDialog()
{
    const char* command =
        "zenity --file-selection --title=\"Select Binary PNG\" "
        "--filename=\"resources/geometry/\" "
        "--file-filter=\"PNG images | *.png\" 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command, "r"), pclose);
    if (!pipe)
        return {};
    char buffer[2048] = {};
    if (!std::fgets(buffer, sizeof(buffer), pipe.get()))
        return {};
    std::string path(buffer);
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
        path.pop_back();
    return path;
}

std::string QueryNvidiaGpuName()
{
    std::unique_ptr<FILE, decltype(&pclose)> pipe(
        popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r"), pclose);
    if (!pipe)
        return {};
    char buffer[256] = {};
    if (!std::fgets(buffer, sizeof(buffer), pipe.get()))
        return {};
    std::string name(buffer);
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r'))
        name.pop_back();
    return name;
}

struct PreviewMask
{
    int width = 0;
    int height = 0;
    int source_width = 0;
    int source_height = 0;
    std::vector<double> values;
};

const PreviewMask& LoadPreviewMask(const std::string& png_path)
{
    static std::unordered_map<std::string, PreviewMask> cache;
    static PreviewMask empty;
    if (png_path.empty())
        return empty;
    auto found = cache.find(png_path);
    if (found != cache.end())
        return found->second;
    PreviewMask mask;
    std::string error;
    LoadBinaryPngMask(png_path, 48, 48, &mask.values, &error, false,
                      &mask.source_width, &mask.source_height);
    mask.width = 48;
    mask.height = 48;
    return cache.emplace(png_path, std::move(mask)).first->second;
}

Rect AspectFit(Rect bounds, int width, int height)
{
    if (width <= 0 || height <= 0 || bounds.w <= 0.0f || bounds.h <= 0.0f)
        return bounds;
    const float image_aspect = static_cast<float>(width) / height;
    const float bounds_aspect = bounds.w / bounds.h;
    if (image_aspect > bounds_aspect)
    {
        const float fitted_height = bounds.w / image_aspect;
        bounds.y += (bounds.h - fitted_height) * 0.5f;
        bounds.h = fitted_height;
    }
    else
    {
        const float fitted_width = bounds.h * image_aspect;
        bounds.x += (bounds.w - fitted_width) * 0.5f;
        bounds.w = fitted_width;
    }
    return bounds;
}

int TopmostNonUniformLayer(const SimulationConfig& config)
{
    for (size_t i = 0; i < config.layers.size(); ++i)
        if (config.layers[i].geometry_mode != "uniform")
            return static_cast<int>(i);
    return -1;
}

Rect PeriodLinkRect() { return {158, 198, 14, 34}; }
Rect OrderLinkRect() { return {158, 256, 14, 34}; }

const char* ZernikeDescription(int index)
{
    static const char* names[37] = {
        "Piston", "Tilt X", "Tilt Y", "Defocus", "Astig 45", "Astig 0",
        "Trefoil Y", "Coma Y", "Coma X", "Trefoil X", "Spherical",
        "2nd Astig 0", "2nd Astig 45", "Quadrafoil 0", "Quadrafoil 45",
        "2nd Coma X", "2nd Coma Y", "2nd Trefoil X", "2nd Trefoil Y",
        "Pentafoil X", "Pentafoil Y", "2nd Spherical", "3rd Astig 45",
        "3rd Astig 0", "3rd Quadrafoil 45", "3rd Quadrafoil 0",
        "Hexafoil Y", "Hexafoil X", "3rd Coma Y", "3rd Coma X",
        "4th Trefoil Y", "4th Trefoil X", "4th Pentafoil Y",
        "4th Pentafoil X", "Heptafoil Y", "Heptafoil X", "3rd Spherical"
    };
    return index >= 0 && index < 37 ? names[index] : "Unknown";
}

void Setup2D(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, width, height, 0.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void DrawButton(const Rect& r, const std::string& label, bool primary = false,
                bool disabled = false, float max_text_scale = 2.0f)
{
    const bool hover = g_app && r.Contains(g_app->mouse_x, g_app->mouse_y) && !disabled;
    const Color fill = disabled ? Color{0.90f,0.92f,0.95f,1} :
                       primary ? (hover ? Color{0.42f,0.70f,0.98f,1} : Color{0.32f,0.60f,0.94f,1}) :
                       (hover ? Color{0.98f,0.99f,1.0f,1} : Color{0.94f,0.97f,1.0f,1});
    const float radius = std::min(9.0f, r.h * 0.35f);
    RoundRect(r, radius, fill);
    RoundOutline(r, radius,
                 primary ? Color{0.20f,0.42f,0.76f,0.95f} : Color{0.72f,0.78f,0.88f,1},
                 hover ? 1.5f : 1.0f);
    float size = max_text_scale;
    while (size > 1.0f && TextWidth(label, size) > r.w - 10.0f)
        size -= 0.1f;
    const float text_w = TextWidth(label, size);
    Text(r.x + (r.w - text_w) * 0.5f, r.y + (r.h - 8.0f * size) * 0.5f,
         label, size, disabled ? Color{0.36f,0.42f,0.47f,1} :
         primary ? Color{1.0f,1.0f,1.0f,1} : Color{0.18f,0.24f,0.32f,1});
}

void Circle(float cx, float cy, float radius, Color color, bool filled = true)
{
    glColor4f(color.r,color.g,color.b,color.a);
    glBegin(filled ? GL_POLYGON : GL_LINE_LOOP);
    for (int i=0;i<48;++i)
    {
        const float a=i*2.0f*kPi/48.0f;
        glVertex2f(cx+std::cos(a)*radius,cy+std::sin(a)*radius);
    }
    glEnd();
}

void DrawSourceIcon(const Rect& r, const std::string& shape, bool selected)
{
    const float cx=r.x+r.w*0.25f;
    const float cy=r.y+r.h*0.5f;
    const Color glow=selected ? Color{0.12f,0.91f,1,1} : Color{0.24f,0.55f,0.66f,1};
    Circle(cx,cy,18,{0.025f,0.055f,0.085f,1});
    Circle(cx,cy,17,{glow.r,glow.g,glow.b,0.14f});
    if (shape=="annular")
    {
        Circle(cx,cy,13,glow);
        Circle(cx,cy,7,{0.025f,0.055f,0.085f,1});
    }
    else if (shape=="dipole")
    {
        Circle(cx-8,cy,6,glow);
        Circle(cx+8,cy,6,glow);
    }
    else if (shape=="quasar")
    {
        Circle(cx-8,cy-8,4.8f,glow); Circle(cx+8,cy-8,4.8f,glow);
        Circle(cx-8,cy+8,4.8f,glow); Circle(cx+8,cy+8,4.8f,glow);
    }
    else
    {
        RoundRect({cx-13,cy-7,26,14},3,glow);
    }
    Circle(r.x+r.w-16,r.y+14,6,selected ? Color{0.12f,0.88f,1,1} : Color{0.12f,0.22f,0.29f,1},false);
    if (selected) Circle(r.x+r.w-16,r.y+14,3,{0.82f,0.99f,1,1});
}

void DrawSourceChoice(const Rect& r, const std::string& shape, const std::string& label, bool selected)
{
    const bool hover=g_app && r.Contains(g_app->mouse_x,g_app->mouse_y);
    RoundRect(r,10,selected ? Color{0.86f,0.94f,1.0f,1} :
              hover ? Color{0.94f,0.97f,1.0f,1} : Color{0.97f,0.98f,1.0f,1});
    RoundOutline(r,10,selected ? Color{0.22f,0.58f,0.92f,1} : Color{0.72f,0.80f,0.90f,1},
                 selected ? 1.6f : 1.0f);
    DrawSourceIcon(r,shape,selected);
    Text(r.x+62,r.y+21,label,1.18f,selected ? Color{0.12f,0.34f,0.58f,1} : Color{0.36f,0.50f,0.64f,1});
}

void DrawField(const NumericField& field, int index, const App& app)
{
    const bool active = index == app.active_field;
    const bool hover = field.enabled && field.rect.Contains(app.mouse_x, app.mouse_y);
    Text(field.rect.x + 2, field.rect.y - 16.0f, field.label, 1.15f, field.enabled ? Color{0.34f,0.44f,0.55f,1} : Color{0.60f,0.66f,0.74f,1});
    Quad(field.rect, !field.enabled ? Color{0.94f,0.95f,0.97f,1} :
         active ? Color{0.95f,0.98f,1.0f,1} :
         hover ? Color{0.97f,0.99f,1.0f,1} : Color{0.985f,0.99f,1.0f,1});
    Outline(field.rect, !field.enabled ? Color{0.84f,0.86f,0.90f,1} :
            active ? Color{0.20f,0.52f,0.82f,1} :
            hover ? Color{0.52f,0.66f,0.80f,1} : Color{0.72f,0.79f,0.88f,1}, active ? 1.8f : 1.0f);
    std::string value = index == app.active_field ? app.edit_buffer :
                        field.int_value ? std::to_string(*field.int_value) : Format(*field.real_value, 2);
    Text(field.rect.x + 10, field.rect.y + 7, value, 1.55f, field.enabled ? Color{0.14f,0.20f,0.28f,1} : Color{0.62f,0.66f,0.72f,1});
}

void DrawTextField(const TextField& field, int index, const App& app)
{
    const bool active = index == app.active_text_field;
    const bool hover = field.rect.Contains(app.mouse_x, app.mouse_y);
    Text(field.rect.x + 2, field.rect.y - 16.0f, field.label, 1.15f, {0.34f,0.44f,0.55f,1});
    Quad(field.rect, active ? Color{0.95f,0.98f,1.0f,1} :
         hover ? Color{0.97f,0.99f,1.0f,1} : Color{0.985f,0.99f,1.0f,1});
    Outline(field.rect, active ? Color{0.20f,0.52f,0.82f,1} :
            hover ? Color{0.52f,0.66f,0.80f,1} : Color{0.72f,0.79f,0.88f,1}, active ? 1.8f : 1.0f);
    const std::string& value = active ? app.edit_buffer : *field.value;
    Text(field.rect.x + 10, field.rect.y + 7, value.empty() ? "path/to/mask.png" : value, 1.2f,
         value.empty() ? Color{0.55f,0.62f,0.72f,1} : Color{0.14f,0.20f,0.28f,1});
}

void DrawRadio(const Rect& r, const std::string& label, bool selected)
{
    const bool hover = g_app && r.Contains(g_app->mouse_x, g_app->mouse_y);
    if (hover || selected)
        RoundRect(r, 8, selected ? Color{0.025f,0.12f,0.18f,1} : Color{0.045f,0.075f,0.11f,1});
    const float cx = r.x + 15.0f;
    const float cy = r.y + r.h * 0.5f;
    glColor4f(selected ? 0.16f : 0.20f, selected ? 0.88f : 0.36f, selected ? 1.0f : 0.45f, 1);
    glLineWidth(selected ? 2.0f : 1.2f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 30; ++i)
    {
        const float angle = i * 2.0f * kPi / 30.0f;
        glVertex2f(cx + std::cos(angle) * 7.0f, cy + std::sin(angle) * 7.0f);
    }
    glEnd();
    if (selected)
    {
        glColor4f(0.12f,0.88f,1,1);
        glBegin(GL_POLYGON);
        for (int i = 0; i < 24; ++i)
        {
            const float angle = i * 2.0f * kPi / 24.0f;
            glVertex2f(cx + std::cos(angle) * 3.5f, cy + std::sin(angle) * 3.5f);
        }
        glEnd();
    }
    Text(r.x + 32, r.y + 9, label, 1.35f,
         selected ? Color{0.84f,0.96f,1,1} : Color{0.48f,0.66f,0.74f,1});
}

void DrawToggle(const Rect& r, const std::string& label, const std::string& off_text,
                const std::string& on_text, bool enabled)
{
    Text(r.x, r.y, label, 1.25f, {0.48f,0.70f,0.78f,1});
    const Rect track{r.x, r.y + 24, r.w, 42};
    const bool hover = g_app && track.Contains(g_app->mouse_x, g_app->mouse_y);
    if (enabled)
        Shadow(track, 12, {0.02f,0.78f,1,hover ? 0.72f : 0.45f});
    RoundRect(track, 12, enabled ? Color{0.025f,0.55f,0.82f,1} :
              hover ? Color{0.075f,0.125f,0.18f,1} : Color{0.045f,0.075f,0.115f,1});
    RoundOutline(track, 12, enabled ? Color{0.24f,0.91f,1,1} : Color{0.16f,0.29f,0.39f,1});
    const float knob_w = track.w * 0.47f;
    const Rect knob{enabled ? track.x + track.w - knob_w - 4.0f : track.x + 4.0f,
                    track.y + 4.0f, knob_w, track.h - 8.0f};
    RoundRect(knob, 9, enabled ? Color{0.88f,0.98f,1,1} : Color{0.18f,0.25f,0.32f,1});
    const std::string active = enabled ? on_text : off_text;
    Text(knob.x + (knob.w - TextWidth(active,1.25f))*0.5f, knob.y + 9, active, 1.25f,
         enabled ? Color{0.02f,0.40f,0.62f,1} : Color{0.68f,0.78f,0.83f,1});
    Text(enabled ? track.x + 10 : track.x + track.w - TextWidth(enabled ? off_text : on_text,0.95f) - 10,
         track.y + 13, enabled ? off_text : on_text, 0.95f,
         enabled ? Color{0.45f,0.78f,0.86f,1} : Color{0.27f,0.48f,0.57f,1});
}

void Cube(float x0, float x1, float y0, float y1, float z0, float z1, Color c)
{
    glColor4f(c.r, c.g, c.b, c.a);
    glBegin(GL_QUADS);
    glNormal3f(0,0,1); glVertex3f(x0,y0,z1); glVertex3f(x1,y0,z1); glVertex3f(x1,y1,z1); glVertex3f(x0,y1,z1);
    glNormal3f(0,0,-1); glVertex3f(x0,y1,z0); glVertex3f(x1,y1,z0); glVertex3f(x1,y0,z0); glVertex3f(x0,y0,z0);
    glNormal3f(0,1,0); glVertex3f(x0,y1,z0); glVertex3f(x0,y1,z1); glVertex3f(x1,y1,z1); glVertex3f(x1,y1,z0);
    glNormal3f(0,-1,0); glVertex3f(x0,y0,z0); glVertex3f(x1,y0,z0); glVertex3f(x1,y0,z1); glVertex3f(x0,y0,z1);
    glNormal3f(1,0,0); glVertex3f(x1,y0,z0); glVertex3f(x1,y1,z0); glVertex3f(x1,y1,z1); glVertex3f(x1,y0,z1);
    glNormal3f(-1,0,0); glVertex3f(x0,y0,z0); glVertex3f(x0,y0,z1); glVertex3f(x0,y1,z1); glVertex3f(x0,y1,z0);
    glEnd();
    glColor4f(c.r + 0.2f, c.g + 0.2f, c.b + 0.2f, 0.8f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP); glVertex3f(x0,y0,z1); glVertex3f(x1,y0,z1); glVertex3f(x1,y1,z1); glVertex3f(x0,y1,z1); glEnd();
}

void DrawExtrudedMask(const PreviewMask& mask,
                      float x0,
                      float x1,
                      float y0,
                      float y1,
                      float z0,
                      float z1,
                      Color color)
{
    if (mask.values.empty() || mask.width <= 0 || mask.height <= 0)
        return;
    const float dx = (x1 - x0) / mask.width;
    const float dy = (y1 - y0) / mask.height;
    for (int iy = 0; iy < mask.height; ++iy)
    {
        int run_begin = -1;
        for (int ix = 0; ix <= mask.width; ++ix)
        {
            const bool filled = ix < mask.width &&
                mask.values[static_cast<size_t>(ix) * mask.height + iy] >= 0.5;
            if (filled && run_begin < 0)
                run_begin = ix;
            if (!filled && run_begin >= 0)
            {
                Cube(x0 + run_begin * dx, x0 + ix * dx,
                     y0 + iy * dy, y0 + (iy + 1) * dy,
                     z0, z1, color);
                run_begin = -1;
            }
        }
    }
}

void DrawStructure3D(const App& app, Rect panel, int window_height)
{
    const int vx = static_cast<int>(panel.x);
    const int vy = window_height - static_cast<int>(panel.y + panel.h);
    const int vw = std::max(1, static_cast<int>(panel.w));
    const int vh = std::max(1, static_cast<int>(panel.h));
    glViewport(vx, vy, vw, vh);
    glEnable(GL_SCISSOR_TEST);
    glScissor(vx, vy, vw, vh);
    glClearColor(0.94f,0.97f,1.0f,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const double aspect = static_cast<double>(vw) / vh;
    glFrustum(-0.65 * aspect, 0.65 * aspect, -0.65, 0.65, 1.2, 30.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, -0.15f, -7.0f * app.camera_zoom);
    float rotation[16];
    QuatMatrix(app.orientation, rotation);
    glMultMatrixf(rotation);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    const GLfloat light[] = {3.0f,-4.0f,8.0f,1.0f};
    const GLfloat ambient[] = {0.18f,0.24f,0.30f,1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

    const double max_period = std::max(app.config.period_x_nm, app.config.period_y_nm);
    const float x_half = max_period > 0.0
        ? 1.55f * static_cast<float>(app.config.period_x_nm / max_period) : 1.55f;
    const float y_half = max_period > 0.0
        ? 1.55f * static_cast<float>(app.config.period_y_nm / max_period) : 1.55f;

    float total = 0.0f;
    for (const LayerConfig& layer : app.config.layers)
        total += static_cast<float>(layer.thickness_nm);
    float z = -0.9f;
    float top_surface_z = z;
    for (auto it = app.config.layers.rbegin(); it != app.config.layers.rend(); ++it)
    {
        const LayerConfig& layer = *it;
        const float h = std::max(0.06f, 1.8f * static_cast<float>(layer.thickness_nm) / std::max(total, 1.0f));
        Color color = MaterialColor(layer.material, layer.color);
        color.a = layer.patterned ? 0.82f : 0.66f;
        if (layer.geometry_mode == "png")
        {
            const PreviewMask& mask = LoadPreviewMask(layer.png_path);
            DrawExtrudedMask(mask, -x_half, x_half, -y_half, y_half, z, z + h, color);
        }
        else
            Cube(-x_half,x_half,-y_half,y_half,z,z+h,color);
        top_surface_z = z + h;
        z += h;
    }

    // The incident plane is coincident with the upper surface of the top layer.
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    const float plane_z = top_surface_z;
    const float plane_x = x_half * 1.08f;
    const float plane_y = y_half * 1.08f;
    glColor4f(0.28f,0.72f,0.96f,0.16f);
    glBegin(GL_QUADS);
    glVertex3f(-plane_x,-plane_y,plane_z);
    glVertex3f( plane_x,-plane_y,plane_z);
    glVertex3f( plane_x, plane_y,plane_z);
    glVertex3f(-plane_x, plane_y,plane_z);
    glEnd();
    glColor4f(0.18f,0.58f,0.88f,0.72f);
    glLineWidth(1.5f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(-plane_x,-plane_y,plane_z);
    glVertex3f( plane_x,-plane_y,plane_z);
    glVertex3f( plane_x, plane_y,plane_z);
    glVertex3f(-plane_x, plane_y,plane_z);
    glEnd();
}

void UploadTexture(GLuint* texture, const std::vector<float>& rgba, int width, int height)
{
    if (rgba.empty())
        return;
    if (!*texture)
        glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, rgba.data());
}

void UploadResultTexture(App* app)
{
    const size_t slice_values = static_cast<size_t>(app->result.width) * app->result.height * 4;
    std::vector<float> farfield_slice;
    if (slice_values > 0 &&
        app->result.farfield_rgba.size() >=
            (static_cast<size_t>(app->selected_focus) + 1) * slice_values)
    {
        const auto begin = app->result.farfield_rgba.begin() +
            static_cast<std::ptrdiff_t>(app->selected_focus * slice_values);
        farfield_slice.assign(begin, begin + static_cast<std::ptrdiff_t>(slice_values));
    }
    UploadTexture(&app->farfield_texture, farfield_slice, app->result.width, app->result.height);
    UploadTexture(&app->nearfield_texture, app->result.nearfield_rgba, app->result.width, app->result.height);
    app->texture_width = app->result.width;
    app->texture_height = app->result.height;
}

bool SaveRgbaPng(const std::string& path, const float* rgba, int width, int height)
{
    if (!rgba || width <= 0 || height <= 0)
        return false;
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file)
        return false;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png ? png_create_info_struct(png) : nullptr;
    if (!png || !info || setjmp(png_jmpbuf(png)))
    {
        if (png)
            png_destroy_write_struct(&png, info ? &info : nullptr);
        std::fclose(file);
        return false;
    }
    png_init_io(png, file);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 4);
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    for (int y = 0; y < height; ++y)
    {
        rows[static_cast<size_t>(y)] = pixels.data() + static_cast<size_t>(y) * width * 4;
        for (int x = 0; x < width; ++x)
            for (int channel = 0; channel < 4; ++channel)
                rows[static_cast<size_t>(y)][x * 4 + channel] =
                    static_cast<unsigned char>(std::clamp(
                        rgba[(static_cast<size_t>(y) * width + x) * 4 + channel],
                        0.0f, 1.0f) * 255.0f + 0.5f);
    }
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(file);
    return true;
}

std::string SavePngDialog(const std::string& suggested_name)
{
    const std::string command =
        "zenity --file-selection --save --confirm-overwrite --title=\"Save PNG\" "
        "--filename=\"" + suggested_name + "\" --file-filter=\"PNG images | *.png\" 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe)
        return {};
    char buffer[2048] = {};
    if (!std::fgets(buffer, sizeof(buffer), pipe.get()))
        return {};
    std::string path(buffer);
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
        path.pop_back();
    if (!path.empty() && std::filesystem::path(path).extension() != ".png")
        path += ".png";
    return path;
}

void DrawImagePanel(const App& app, Rect r, GLuint texture, const std::string& empty_title, const std::string& empty_hint)
{
    GradientQuad(r, {0.96f,0.98f,1.0f,1}, {0.92f,0.96f,1.0f,1});
    if (!texture)
    {
        const std::string line = app.running ? "CUDA SIMULATION RUNNING" : empty_title;
        const float cx = r.x + r.w * 0.5f;
        const float cy = r.y + r.h * 0.5f;
        const float pulse = 0.5f + 0.5f * std::sin(static_cast<float>(glfwGetTime()) * 2.4f);
        glColor4f(0.08f,0.76f,0.96f,0.16f + pulse * 0.12f);
        glLineWidth(2.0f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 48; ++i)
        {
            const float a = i * 2.0f * kPi / 48.0f;
            glVertex2f(cx + std::cos(a) * 35.0f, cy - 28.0f + std::sin(a) * 35.0f);
        }
        glEnd();
        glColor4f(0.18f,0.90f,1.0f,0.85f);
        glBegin(GL_LINES);
        glVertex2f(cx-13,cy-28); glVertex2f(cx+13,cy-28);
        glVertex2f(cx,cy-41); glVertex2f(cx,cy-15);
        glEnd();
        Text(cx - TextWidth(line,1.65f)*0.5f, cy + 28, line, 1.65f, {0.45f,0.72f,0.82f,1});
        const std::string hint = app.running ? "ISOLATED GPU WORKER" : empty_hint;
        Text(cx - TextWidth(hint,1.05f)*0.5f, cy + 50, hint, 1.05f, {0.24f,0.42f,0.52f,1});
        return;
    }
    const float image_aspect = static_cast<float>(app.texture_width) / app.texture_height;
    const float box_aspect = r.w / r.h;
    Rect dst = r;
    if (image_aspect > box_aspect)
    {
        dst.h = r.w / image_aspect;
        dst.y += (r.h - dst.h) * 0.5f;
    }
    else
    {
        dst.w = r.h * image_aspect;
        dst.x += (r.w - dst.w) * 0.5f;
    }
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glColor4f(1,1,1,1);
    glBegin(GL_QUADS);
    glTexCoord2f(0,1); glVertex2f(dst.x,dst.y);
    glTexCoord2f(1,1); glVertex2f(dst.x+dst.w,dst.y);
    glTexCoord2f(1,0); glVertex2f(dst.x+dst.w,dst.y+dst.h);
    glTexCoord2f(0,0); glVertex2f(dst.x,dst.y+dst.h);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

Rect GeometryCanvasRect()
{
    return {22, 132, 286, 142};
}

void DrawGeometryCanvas(const App& app, Rect r)
{
    GradientQuad(r, {0.95f,0.98f,1.0f,1}, {0.90f,0.94f,0.99f,1});
    RoundOutline(r, 10, {0.70f,0.79f,0.90f,1});
    const int layer_index = TopmostNonUniformLayer(app.config);
    if (layer_index < 0)
    {
        Text(r.x + 52, r.y + r.h * 0.5f - 5, "NO NON-UNIFORM LAYER", 1.15f,
             {0.42f,0.54f,0.66f,1});
        return;
    }
    const LayerConfig& layer = app.config.layers[static_cast<size_t>(layer_index)];
    if (layer.geometry_mode == "png")
    {
        const PreviewMask& mask = LoadPreviewMask(layer.png_path);
        const Rect fitted = AspectFit({r.x + 8.0f, r.y + 8.0f, r.w - 16.0f, r.h - 16.0f},
                                      mask.source_width, mask.source_height);
        const float dx = fitted.w / std::max(1, mask.width);
        const float dy = fitted.h / std::max(1, mask.height);
        glColor4f(0.08f, 0.08f, 0.08f, 0.95f);
        glBegin(GL_QUADS);
        for (int ix = 0; ix < mask.width; ++ix)
        {
            for (int iy = 0; iy < mask.height; ++iy)
            {
                if (mask.values.empty() || mask.values[static_cast<size_t>(ix) * mask.height + iy] < 0.5)
                    continue;
                const float xa = fitted.x + ix * dx;
                const float xb = xa + dx;
                const float ya = fitted.y + iy * dy;
                const float yb = ya + dy;
                glVertex2f(xa, ya);
                glVertex2f(xb, ya);
                glVertex2f(xb, yb);
                glVertex2f(xa, yb);
            }
        }
        glEnd();
    }
}

void CommitEdit(App* app)
{
    if (app->active_geom_thickness_row >= 0 &&
        app->active_geom_thickness_row < static_cast<int>(app->config.layers.size()))
    {
        char* end = nullptr;
        const double value = std::strtod(app->edit_buffer.c_str(), &end);
        if (end != app->edit_buffer.c_str() && std::isfinite(value) && value > 0.0)
            app->config.layers[static_cast<size_t>(app->active_geom_thickness_row)].thickness_nm = value;
        app->active_geom_thickness_row = -1;
        app->edit_buffer.clear();
        app->replace_numeric_input = false;
        return;
    }
    if (app->active_text_field >= 0 && app->active_text_field < static_cast<int>(app->text_fields.size()))
    {
        *app->text_fields[app->active_text_field].value = app->edit_buffer;
        app->active_text_field = -1;
        app->edit_buffer.clear();
        return;
    }
    if (app->active_field < 0 || app->active_field >= static_cast<int>(app->fields.size()))
        return;
    NumericField& field = app->fields[app->active_field];
    char* end = nullptr;
    const double value = std::strtod(app->edit_buffer.c_str(), &end);
    if (end != app->edit_buffer.c_str() && std::isfinite(value))
    {
        if (field.int_value) *field.int_value = static_cast<int>(std::lround(value));
        if (field.real_value) *field.real_value = value;
    }
    app->active_field = -1;
    app->edit_buffer.clear();
    app->replace_numeric_input = false;
    if (app->period_xy_linked)
        app->config.period_y_nm = app->config.period_x_nm;
    if (app->order_xy_linked)
        app->config.order_y = app->config.order_x;
    UpdateNyquistGrid(&app->config);
}

void StartEdit(App* app, int index)
{
    CommitEdit(app);
    app->active_field = index;
    const NumericField& field = app->fields[index];
    app->edit_buffer = field.int_value ? std::to_string(*field.int_value) : Format(*field.real_value, 2);
    app->replace_numeric_input = true;
}

void StartTextEdit(App* app, int index)
{
    CommitEdit(app);
    app->active_text_field = index;
    app->edit_buffer = *app->text_fields[index].value;
    app->replace_numeric_input = false;
}

void StartGeomThicknessEdit(App* app, int row)
{
    CommitEdit(app);
    app->active_geom_thickness_row = row;
    app->edit_buffer = Format(app->config.layers[static_cast<size_t>(row)].thickness_nm, 2);
    app->replace_numeric_input = true;
}

void KeyCallback(GLFWwindow*, int key, int, int action, int mods)
{
    if (!g_app || (action != GLFW_PRESS && action != GLFW_REPEAT))
        return;
    if (key == GLFW_KEY_ESCAPE)
    {
        if (g_app->active_field >= 0 || g_app->active_text_field >= 0 || g_app->active_geom_thickness_row >= 0)
        {
            g_app->active_field = -1;
            g_app->active_text_field = -1;
            g_app->active_geom_thickness_row = -1;
            g_app->replace_numeric_input = false;
        }
        else
            glfwSetWindowShouldClose(g_app->window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER || key == GLFW_KEY_TAB)
        CommitEdit(g_app);
    else if (key == GLFW_KEY_BACKSPACE &&
             (g_app->active_field >= 0 || g_app->active_text_field >= 0 || g_app->active_geom_thickness_row >= 0))
    {
        if (g_app->replace_numeric_input)
        {
            g_app->edit_buffer.clear();
            g_app->replace_numeric_input = false;
        }
        else if (!g_app->edit_buffer.empty())
            g_app->edit_buffer.pop_back();
    }
    else if ((mods & GLFW_MOD_CONTROL) && key == GLFW_KEY_S)
    {
        CommitEdit(g_app);
        std::string error;
        g_app->status = SaveSimulationConfig(g_app->json_path, g_app->config, &error) ? "JSON SAVED" : error;
    }
    else if (key == GLFW_KEY_R)
    {
        g_app->orientation = Multiply(AxisAngle({1,0,0}, 38.0f * kPi / 180.0f),
                                      AxisAngle({0,0,1}, -28.0f * kPi / 180.0f));
        g_app->spin = {};
        g_app->camera_zoom = 1.0f;
    }
}

void CharCallback(GLFWwindow*, unsigned int codepoint)
{
    if (!g_app || (g_app->active_field < 0 && g_app->active_text_field < 0 && g_app->active_geom_thickness_row < 0) || codepoint > 127)
        return;
    const char c = static_cast<char>(codepoint);
    if (g_app->active_text_field >= 0)
    {
        if (c >= 32 && c < 127)
            g_app->edit_buffer.push_back(c);
    }
    else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
    {
        if (g_app->replace_numeric_input)
        {
            g_app->edit_buffer.clear();
            g_app->replace_numeric_input = false;
        }
        g_app->edit_buffer.push_back(c);
    }
}

void ScrollCallback(GLFWwindow*, double, double y)
{
    if (!g_app)
        return;
    if (g_app->material_menu_open)
    {
        constexpr int visible_items = 12;
        const int item_count = static_cast<int>(SharedMaterialDatabase().materials().size()) + 1;
        const int maximum = std::max(0, item_count - visible_items);
        g_app->material_menu_scroll = std::clamp(
            g_app->material_menu_scroll - static_cast<int>(y * 3.0), 0, maximum);
        return;
    }
    g_app->camera_zoom = std::clamp(g_app->camera_zoom - static_cast<float>(y) * 0.06f, 0.55f, 1.8f);
}

void BuildFields(App* app, float panel_x, float panel_y, float panel_w)
{
    app->fields.clear();
    app->text_fields.clear();
    const float gap = 10.0f;
    const float w = (panel_w - gap) * 0.5f;
    const float h = 34.0f;
    auto add_real = [&](const char* label, double* value, int col, int row, bool enabled = true) {
        app->fields.push_back({label, value, nullptr, {panel_x + col * (w + gap), panel_y + row * 58.0f, w, h}, enabled});
    };
    auto add_int = [&](const char* label, int* value, int col, int row, bool enabled = true) {
        app->fields.push_back({label, nullptr, value, {panel_x + col * (w + gap), panel_y + row * 58.0f, w, h}, enabled});
    };
    if (app->active_tab == kBasicTab)
    {
        add_real("WAVELENGTH NM", &app->config.wavelength_nm, 0, 0);
        add_int("IMAGE WIDTH", &app->config.image_width, 1, 0);
        add_real("PERIOD X NM", &app->config.period_x_nm, 0, 1);
        add_real("PERIOD Y NM", &app->config.period_y_nm, 1, 1, !app->period_xy_linked);
        add_int("RCWA ORDER X", &app->config.order_x, 0, 2);
        add_int("RCWA ORDER Y", &app->config.order_y, 1, 2, !app->order_xy_linked);
        app->selected_focus = std::clamp(app->selected_focus, 0,
            std::max(0, static_cast<int>(app->config.focus_nm.size()) - 1));
        add_real("FOCUS NM", &app->config.focus_nm[static_cast<size_t>(app->selected_focus)], 0, 3);
    }
    else if (app->active_tab == kSourceTab)
    {
        const float source_y=292.0f;
        app->fields.push_back({"SIGMA OUT",&app->config.sigma_outer,nullptr,{panel_x,source_y,w,h}});
        app->fields.push_back({"SIGMA IN",&app->config.sigma_inner,nullptr,{panel_x+w+gap,source_y,w,h}});
        const bool dipole = app->config.source_shape == "dipole";
        const bool quasar = app->config.source_shape == "quasar";
        const bool rect = app->config.source_shape == "rect";
        app->fields[0].enabled = !rect;
        app->fields[1].enabled = !rect;
        app->fields.push_back({"OPEN ANGLE",&app->config.source_open_angle_deg,nullptr,
                               {panel_x,source_y+58,w,h},dipole});
        app->fields.push_back({"ROTATE ANGLE",&app->config.source_rotate_angle_deg,nullptr,
                               {panel_x+w+gap,source_y+58,w,h},dipole || quasar || rect});
        app->fields.push_back({"RECT X",&app->config.source_rect_x,nullptr,
                               {panel_x,source_y+116,w,h},rect});
        app->fields.push_back({"RECT Y",&app->config.source_rect_y,nullptr,
                               {panel_x+w+gap,source_y+116,w,h},rect});
    }
    else if (app->active_tab == kPolarizationTab)
    {
        if (app->config.polarization == "angle")
            add_real("POLARIZATION DEG", &app->config.polarization_angle_deg, 0, 4);
    }
    else if (app->active_tab == kZernikeTab)
    {
        const int begin = app->zernike_page * 19;
        for (int i = 0; i < 19 && begin + i < 37; ++i)
            add_real(("Z" + std::to_string(begin + i) + " " + ZernikeDescription(begin + i)).c_str(),
                     &app->config.zernike[begin + i], i % 2, i / 2);
    }
}

void StopSimulation(App* app)
{
    if (!app->running || app->worker_pid <= 0)
        return;
    kill(app->worker_pid, SIGKILL);
    waitpid(app->worker_pid, nullptr, 0);
    app->worker_pid = -1;
    app->running = false;
    app->status = "SIMULATION FORCE STOPPED";
}

void StartSimulation(App* app)
{
    if (app->running)
        return;
    CommitEdit(app);
    std::string error;
    if (!ValidateSimulationConfig(app->config, &error))
    {
        app->status = error;
        return;
    }
    app->config.incident_theta_deg = 0.0;
    app->config.incident_phi_deg = 0.0;
    UpdateNyquistGrid(&app->config);
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    app->worker_config_path = (temp_dir / ("worker_" + std::to_string(getpid()) + ".json")).string();
    app->worker_result_path = (temp_dir / ("worker_" + std::to_string(getpid()) + ".bin")).string();
    if (!SaveSimulationConfig(app->worker_config_path, app->config, &error))
    {
        app->status = error;
        return;
    }
    std::filesystem::remove(app->worker_result_path);
    const std::string worker = app->executable_dir +
        (app->config.precision == "double" ? "/worker_fp64" : "/worker_fp32");
    const pid_t pid = fork();
    if (pid == 0)
    {
        execl(worker.c_str(), worker.c_str(), app->worker_config_path.c_str(),
              app->worker_result_path.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    if (pid < 0)
    {
        app->status = "FAILED TO START CUDA WORKER";
        return;
    }
    app->worker_pid = pid;
    app->running = true;
    app->status = "CUDA " + app->config.precision + " / APPROXIMATION " +
                  std::to_string(app->config.approximation_order) + " RUNNING";
}

void HandleClick(App* app, int width, int height)
{
    const float left = 330.0f;
    const Rect run{22, height - 68.0f, 286, 44};
    const Rect load{22, height - 122.0f, 89, 36};
    const Rect save{119, height - 122.0f, 89, 36};
    const float content_w = width - left - 54;
    const Rect structure{left + 18, 82, content_w * 0.48f,
                         std::max(220.0f, height - 136.0f)};
    const float image_height = (structure.h - 18.0f) * 0.5f;
    const Rect image_far{structure.x + structure.w + 18, 82,
                         content_w - structure.w - 18, image_height};
    const Rect image_near{image_far.x, image_far.y + image_far.h + 18.0f,
                          image_far.w, image_height};
    app->structure_panel = structure;
    const float tab_w = 47.5f;
    for (int i = 0; i < 6; ++i)
        if (Rect{22 + i * tab_w, 78, tab_w - 3, 30}.Contains(app->mouse_x, app->mouse_y))
        {
            CommitEdit(app);
            app->active_tab = i;
            return;
        }

    for (int i = 0; i < static_cast<int>(app->fields.size()); ++i)
        if (app->fields[i].enabled && app->fields[i].rect.Contains(app->mouse_x, app->mouse_y))
        {
            StartEdit(app, i);
            return;
        }
    for (int i = 0; i < static_cast<int>(app->text_fields.size()); ++i)
        if (app->text_fields[i].rect.Contains(app->mouse_x, app->mouse_y))
        {
            StartTextEdit(app, i);
            return;
        }
    CommitEdit(app);
    if (Rect{image_far.x + image_far.w - 172, image_far.y + 7, 24, 22}.Contains(app->mouse_x, app->mouse_y) &&
        app->result.focus_count > 0)
    {
        app->selected_focus = (app->selected_focus + app->result.focus_count - 1) % app->result.focus_count;
        UploadResultTexture(app);
        return;
    }
    if (Rect{image_far.x + image_far.w - 76, image_far.y + 7, 24, 22}.Contains(app->mouse_x, app->mouse_y) &&
        app->result.focus_count > 0)
    {
        app->selected_focus = (app->selected_focus + 1) % app->result.focus_count;
        UploadResultTexture(app);
        return;
    }
    if (Rect{image_far.x + image_far.w - 48, image_far.y + 7, 42, 22}.Contains(app->mouse_x, app->mouse_y))
    {
        const size_t slice_values = static_cast<size_t>(app->result.width) * app->result.height * 4;
        if (slice_values && app->result.farfield_rgba.size() >=
                (static_cast<size_t>(app->selected_focus) + 1) * slice_values)
        {
            const std::string path = SavePngDialog("farfield.png");
            const float* slice = app->result.farfield_rgba.data() +
                static_cast<size_t>(app->selected_focus) * slice_values;
            app->status = !path.empty() && SaveRgbaPng(path, slice, app->result.width, app->result.height)
                ? "FAR-FIELD IMAGE SAVED" : "FAR-FIELD SAVE CANCELLED";
        }
        return;
    }
    if (Rect{image_near.x + image_near.w - 48, image_near.y + 7, 42, 22}.Contains(app->mouse_x, app->mouse_y))
    {
        const std::string path = SavePngDialog("nearfield.png");
        app->status = !path.empty() && SaveRgbaPng(path, app->result.nearfield_rgba.data(),
                                                   app->result.width, app->result.height)
            ? "NEAR-FIELD IMAGE SAVED" : "NEAR-FIELD SAVE CANCELLED";
        return;
    }
    if (structure.Contains(app->mouse_x, app->mouse_y))
    {
        app->dragging_view = true;
        app->spin = {};
        return;
    }
    if (run.Contains(app->mouse_x, app->mouse_y))
    {
        if (app->running) StopSimulation(app); else StartSimulation(app);
    }
    else if (load.Contains(app->mouse_x, app->mouse_y))
    {
        std::string error;
        app->status = LoadSimulationConfig(app->json_path, &app->config, &error) ? "JSON LOADED" : error;
        app->selected_layer = std::clamp(app->selected_layer, 0, std::max(0, static_cast<int>(app->config.layers.size()) - 1));
    }
    else if (save.Contains(app->mouse_x, app->mouse_y))
    {
        std::string error;
        app->status = SaveSimulationConfig(app->json_path, app->config, &error) ? "JSON SAVED" : error;
    }
    else if (app->active_tab == kBasicTab && PeriodLinkRect().Contains(app->mouse_x, app->mouse_y))
    {
        app->period_xy_linked = !app->period_xy_linked;
        if (app->period_xy_linked)
            app->config.period_y_nm = app->config.period_x_nm;
        return;
    }
    else if (app->active_tab == kBasicTab && OrderLinkRect().Contains(app->mouse_x, app->mouse_y))
    {
        app->order_xy_linked = !app->order_xy_linked;
        if (app->order_xy_linked)
            app->config.order_y = app->config.order_x;
        return;
    }
    else if (app->active_tab == kBasicTab && Rect{170,314,62,34}.Contains(app->mouse_x, app->mouse_y))
    {
        const double next_focus = app->config.focus_nm.empty() ? 0.0 : app->config.focus_nm.back() + 100.0;
        app->config.focus_nm.push_back(next_focus);
        app->selected_focus = static_cast<int>(app->config.focus_nm.size()) - 1;
        return;
    }
    else if (app->active_tab == kBasicTab && Rect{240,314,62,34}.Contains(app->mouse_x, app->mouse_y))
    {
        if (app->config.focus_nm.size() > 1)
        {
            app->config.focus_nm.erase(app->config.focus_nm.begin() + app->selected_focus);
            app->selected_focus = std::clamp(app->selected_focus, 0,
                static_cast<int>(app->config.focus_nm.size()) - 1);
        }
        return;
    }
    else if (app->active_tab == kSourceTab)
    {
        static const char* shapes[]={"annular","dipole","quasar","rect"};
        for (int i=0;i<4;++i)
        {
            const Rect card{28.0f+(i%2)*138.0f,142.0f+(i/2)*72.0f,128,62};
            if (card.Contains(app->mouse_x,app->mouse_y))
            {
                app->config.source_shape=shapes[i];
                return;
            }
        }
    }
    else if (app->active_tab == kPolarizationTab)
    {
        static const char* polarizations[] = {"x","y","circular","angle"};
        for (int i = 0; i < 4; ++i)
            if (Rect{28,146.0f + i * 42.0f,272,36}.Contains(app->mouse_x,app->mouse_y))
            {
                app->config.polarization = polarizations[i];
                return;
            }
    }
    else if (app->active_tab == kZernikeTab && Rect{22,height-182.0f,138,32}.Contains(app->mouse_x, app->mouse_y))
        app->zernike_page = std::max(0, app->zernike_page - 1);
    else if (app->active_tab == kZernikeTab && Rect{170,height-182.0f,138,32}.Contains(app->mouse_x, app->mouse_y))
        app->zernike_page = std::min(1, app->zernike_page + 1);
    else if (app->active_tab == kGeometryTab)
    {
        const float row_y0 = 326.0f;
        const int visible_layers = std::max(
            1, std::min(static_cast<int>(app->config.layers.size()),
                        static_cast<int>((height - 164.0f - row_y0) / 27.0f) + 1));
        if (app->material_menu_open)
        {
            std::vector<std::string> names{"custom"};
            for (const MaterialRecord& record : SharedMaterialDatabase().materials())
                names.push_back(record.name);
            constexpr int visible_items = 12;
            const float menu_y = 158.0f;
            for (int slot = 0; slot < visible_items; ++slot)
            {
                const int index = app->material_menu_scroll + slot;
                if (index >= static_cast<int>(names.size()))
                    break;
                Rect item{22, menu_y + static_cast<float>(slot) * 24.0f, 286, 22};
                if (item.Contains(app->mouse_x, app->mouse_y))
                {
                    LayerConfig& layer = app->config.layers[static_cast<size_t>(app->material_menu_row)];
                    layer.material = names[static_cast<size_t>(index)];
                    if (layer.material != "custom")
                    {
                        SharedMaterialDatabase().Lookup(layer.material, app->config.wavelength_nm, &layer.n_real, &layer.n_imag);
                        const Color color = MaterialColor(layer.material, layer.color);
                        layer.color = {color.r, color.g, color.b};
                    }
                    app->material_menu_open = false;
                    return;
                }
            }
            app->material_menu_open = false;
        }
        for (int i = 0; i < visible_layers; ++i)
        {
            const float y = row_y0 + static_cast<float>(i) * 27.0f;
            if (Rect{22, y, 94, 24}.Contains(app->mouse_x, app->mouse_y))
            {
                app->selected_layer = static_cast<int>(i);
                app->material_menu_open = true;
                app->material_menu_row = static_cast<int>(i);
                app->material_menu_scroll = 0;
                const std::string& current = app->config.layers[static_cast<size_t>(i)].material;
                const auto& records = SharedMaterialDatabase().materials();
                for (size_t material_index = 0; material_index < records.size(); ++material_index)
                    if (records[material_index].name == current)
                    {
                        app->material_menu_scroll =
                            std::max(0, static_cast<int>(material_index) + 1 - 5);
                        break;
                    }
                return;
            }
            if (Rect{120, y + 1, 62, 22}.Contains(app->mouse_x, app->mouse_y))
            {
                app->selected_layer = static_cast<int>(i);
                StartGeomThicknessEdit(app, static_cast<int>(i));
                return;
            }
            if (Rect{186, y + 1, 88, 22}.Contains(app->mouse_x, app->mouse_y))
            {
                app->selected_layer = static_cast<int>(i);
                LayerConfig& layer = app->config.layers[static_cast<size_t>(i)];
                if (layer.geometry_mode == "uniform")
                {
                    layer.geometry_mode = "png";
                    layer.patterned = true;
                    if (layer.png_path.empty())
                        layer.png_path = "resources/geometry/DSW-c.png";
                }
                else
                {
                    layer.geometry_mode = "uniform";
                    layer.patterned = false;
                }
                return;
            }
            if (Rect{280, y, 12, 11}.Contains(app->mouse_x, app->mouse_y))
            {
                LayerConfig layer;
                layer.name = "Layer " + std::to_string(app->config.layers.size() + 1);
                layer.material = "Si";
                layer.thickness_nm = 100.0;
                layer.n_real = 1.0;
                layer.n_imag = 0.0;
                SharedMaterialDatabase().Lookup("Si", app->config.wavelength_nm,
                                                &layer.n_real, &layer.n_imag);
                layer.patterned = false;
                layer.geometry_mode = "uniform";
                layer.png_path.clear();
                const Color silicon_color = MaterialColor("Si", layer.color);
                layer.color = {silicon_color.r, silicon_color.g, silicon_color.b};
                app->config.layers.insert(app->config.layers.begin() + static_cast<long>(i + 1), layer);
                app->selected_layer = static_cast<int>(i + 1);
                return;
            }
            if (Rect{280, y + 13, 12, 11}.Contains(app->mouse_x, app->mouse_y))
            {
                if (app->config.layers.size() > 1)
                {
                    app->config.layers.erase(app->config.layers.begin() + static_cast<long>(i));
                    app->selected_layer = std::clamp(app->selected_layer, 0, static_cast<int>(app->config.layers.size()) - 1);
                }
                return;
            }
        }
        if (Rect{216,282,92,30}.Contains(app->mouse_x, app->mouse_y))
        {
            const std::string path = OpenPngDialog();
            if (!path.empty() && !app->config.layers.empty())
            {
                LayerConfig& layer = app->config.layers[app->selected_layer];
                layer.png_path = path;
                layer.geometry_mode = "png";
                layer.patterned = true;
                app->status = "PNG MASK SELECTED";
            }
            return;
        }
    }
    else if (app->active_tab == kSolverTab && Rect{22,154,286,42}.Contains(app->mouse_x, app->mouse_y))
        app->config.precision = app->config.precision == "double" ? "single" : "double";
    else if (app->active_tab == kSolverTab && Rect{22,254,286,42}.Contains(app->mouse_x, app->mouse_y))
        app->config.approximation_order = app->config.approximation_order == 12 ? 18 : 12;
}

void DrawUi(App* app, int width, int height)
{
    const Color cyan{0.26f,0.56f,0.88f,1};
    const float left = 330.0f;
    const float content_w = width - left - 54;
    const float panel_height = std::max(220.0f, height - 136.0f);
    const Rect structure{left + 18, 82, content_w * 0.48f, panel_height};
    const float image_height = (panel_height - 18.0f) * 0.5f;
    const Rect image_far{structure.x + structure.w + 18, 82,
                         content_w - structure.w - 18, image_height};
    const Rect image_near{image_far.x, image_far.y + image_far.h + 18.0f,
                          image_far.w, image_height};
    app->structure_panel = structure;

    Setup2D(width, height);
    GradientQuad({0,0,static_cast<float>(width),static_cast<float>(height)},
                 {0.96f,0.98f,1.0f,1}, {0.90f,0.94f,0.99f,1});
    glColor4f(0.58f,0.70f,0.86f,0.18f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int x = static_cast<int>(left); x < width; x += 42)
    {
        glVertex2f(static_cast<float>(x),0); glVertex2f(static_cast<float>(x),static_cast<float>(height));
    }
    for (int y = 0; y < height; y += 42)
    {
        glVertex2f(left,static_cast<float>(y)); glVertex2f(static_cast<float>(width),static_cast<float>(y));
    }
    glEnd();
    GradientQuad({0,0,left,static_cast<float>(height)},
                 {0.93f,0.96f,1.0f,0.99f}, {0.86f,0.91f,0.98f,0.99f});
    GlowLine(left,0,left,height,cyan);

    Shadow(structure, 14, {0.40f,0.52f,0.70f,0.32f});
    Shadow(image_far, 14, {0.40f,0.52f,0.70f,0.32f});
    Shadow(image_near, 14, {0.40f,0.52f,0.70f,0.32f});
    RoundRect(structure, 14, {0.98f,0.99f,1.0f,1});
    RoundRect(image_far, 14, {0.98f,0.99f,1.0f,1});
    RoundRect(image_near, 14, {0.98f,0.99f,1.0f,1});
    DrawStructure3D(*app, structure, height);
    Setup2D(width, height);

    RoundRect({20,18,46,46},13,{0.30f,0.55f,0.88f,1});
    RoundOutline({20,18,46,46},13,{0.18f,0.37f,0.66f,1},1.5f);
    const float s_scale = 4.0f;
    Text(43.0f - TextWidth("S", s_scale) * 0.5f, 26.0f, "S", s_scale, {0.99f, 1, 1, 1});
    float title_scale = 3.0f;
    while (title_scale > 2.0f && 79.0f + TextWidth("SIMULATOR", title_scale) > 220.0f)
        title_scale -= 0.1f;
    Text(79, 18, "SIMULATOR", title_scale, {0.12f,0.18f,0.26f,1});
    Text(80, 48, "GPU OPTICAL WORKBENCH", 1.15f, {0.38f,0.50f,0.62f,1});
    RoundRect({230,24,78,25},12,{0.89f,0.94f,1.0f,1});
    RoundRect({239,33,7,7},4,app->running ? Color{1.0f,0.46f,0.20f,1} : Color{0.22f,0.95f,0.62f,1});
    Text(252,30,app->running ? "BUSY" : "READY",1.05f,{0.28f,0.42f,0.58f,1});

    Text(left + 22, 22, "Multilayer Structure", 1.85f, {0.18f,0.24f,0.32f,1});
    Text(image_far.x, 22, "Farfield / Nearfield", 1.85f, {0.18f,0.24f,0.32f,1});
    RoundRect({image_far.x + image_far.w - 108,22,108,25},12,{0.90f,0.95f,1.0f,0.95f});
    Text(image_far.x + image_far.w - 96,29,app->config.precision+" / F"+std::to_string(app->config.approximation_order),
         1.05f,{0.28f,0.45f,0.68f,1});

    static const char* tabs[] = {"LAYERS","MODEL","SOURCE","POLAR.","ABERR.","SOLVER"};
    const Color tab_panel{0.94f,0.97f,1.0f,0.98f};
    const Rect control_panel{18,106,294,std::max(354.0f,height-264.0f)};
    RoundRect(control_panel,12,tab_panel);
    RoundOutline(control_panel,12,{0.72f,0.80f,0.90f,0.9f});
    RoundRect({18,74,294,38},11,{0.87f,0.92f,0.98f,1});
    for (int i = 0; i < 6; ++i)
    {
        const Rect tab{22 + i * 47.5f,78,45,30};
        if (app->active_tab == i)
        {
            RoundRect(tab,8,tab_panel);
            Quad({tab.x,tab.y+20,tab.w,12},tab_panel);
            RoundOutline(tab,8,{0.28f,0.58f,0.90f,1},1.5f);
            Quad({tab.x+1,tab.y+22,tab.w-2,10},tab_panel);
        }
        else if (tab.Contains(app->mouse_x,app->mouse_y))
            RoundRect(tab,8,{0.93f,0.96f,1.0f,1});
        float tab_size = 1.0f;
        while (TextWidth(tabs[i],tab_size) > tab.w - 6 && tab_size > 0.72f) tab_size -= 0.05f;
        Text(tab.x+(tab.w-TextWidth(tabs[i],tab_size))*0.5f,tab.y+10,tabs[i],tab_size,
             app->active_tab == i ? Color{0.16f,0.42f,0.72f,1} : Color{0.38f,0.50f,0.62f,1});
    }

    BuildFields(app, 22, 140, 286);
    for (int i = 0; i < static_cast<int>(app->fields.size()); ++i)
        DrawField(app->fields[i], i, *app);
    for (int i = 0; i < static_cast<int>(app->text_fields.size()); ++i)
        DrawTextField(app->text_fields[i], i, *app);

    if (app->active_tab == kBasicTab)
    {
        UpdateNyquistGrid(&app->config);
        if (app->period_xy_linked)
            app->config.period_y_nm = app->config.period_x_nm;
        if (app->order_xy_linked)
            app->config.order_y = app->config.order_x;
        DrawButton(PeriodLinkRect(), "=", true);
        DrawButton(OrderLinkRect(), "=", true);
        DrawButton({170,314,62,34},"+ FOCUS",false,false,0.95f);
        DrawButton({240,314,62,34},"- FOCUS",false,app->config.focus_nm.size() <= 1,0.95f);
        Text(170,356,"FOCUS "+std::to_string(app->selected_focus+1)+" / "+
             std::to_string(app->config.focus_nm.size()),0.95f,{0.34f,0.50f,0.66f,1});
    }
    else if (app->active_tab == kSourceTab)
    {
        Text(28,124,"ILLUMINATION APERTURE",1.08f,{0.38f,0.67f,0.76f,1});
        static const char* shapes[]={"annular","dipole","quasar","rect"};
        static const char* labels[]={"ANNULAR","DIPOLE","QUASAR","RECT"};
        for (int i=0;i<4;++i)
            DrawSourceChoice({28.0f+(i%2)*138.0f,142.0f+(i/2)*72.0f,128,62},
                             shapes[i],labels[i],app->config.source_shape==shapes[i]);
    }
    else if (app->active_tab == kPolarizationTab)
    {
        Text(28,128,"POLARIZATION",1.15f,{0.38f,0.67f,0.76f,1});
        static const char* values[] = {"x","y","circular","angle"};
        static const char* labels[] = {"LINEAR X","LINEAR Y","CIRCULAR","CUSTOM ANGLE"};
        for (int i = 0; i < 4; ++i)
            DrawRadio({28,146.0f+i*42.0f,272,36},labels[i],app->config.polarization == values[i]);
    }
    else if (app->active_tab == kZernikeTab)
    {
        DrawButton({22,height-182.0f,138,32},"PREVIOUS");
        DrawButton({170,height-182.0f,138,32},"NEXT");
        Text(22,height-144.0f,"PAGE "+std::to_string(app->zernike_page+1)+" / 2",
             1.2f,{0.38f,0.67f,0.76f,1});
    }
    else if (app->active_tab == kGeometryTab)
    {
        const int preview_index = TopmostNonUniformLayer(app->config);
        DrawGeometryCanvas(*app, GeometryCanvasRect());
        if (preview_index >= 0)
        {
            const LayerConfig& preview_layer = app->config.layers[static_cast<size_t>(preview_index)];
            Text(32,160,preview_layer.name,0.95f,{0.42f,0.54f,0.66f,1});
            Text(22,288,preview_layer.png_path.empty() ? "NO PNG SELECTED" :
                 std::filesystem::path(preview_layer.png_path).filename().string(),
                 0.90f,{0.34f,0.44f,0.58f,1});
        }
        else
            Text(22,288,"SET A LAYER TO NON-UNIFORM",0.90f,{0.34f,0.44f,0.58f,1});
        DrawButton({216,282,92,30},"BROWSE",false,false,1.25f);
        Text(24,310,"LAYER",0.90f,{0.34f,0.48f,0.62f,1});
        Text(120,310,"THICK",0.90f,{0.34f,0.48f,0.62f,1});
        Text(186,310,"GEOMETRY",0.90f,{0.34f,0.48f,0.62f,1});
        const float row_y0 = 326.0f;
        const int visible_layers = std::max(
            1, std::min(static_cast<int>(app->config.layers.size()),
                        static_cast<int>((height - 164.0f - row_y0) / 27.0f) + 1));
        for (int i = 0; i < visible_layers; ++i)
        {
            const LayerConfig& layer = app->config.layers[static_cast<size_t>(i)];
            const float y = row_y0 + static_cast<float>(i) * 27.0f;
            const bool selected = static_cast<int>(i) == app->selected_layer;
            RoundRect({22, y, 286, 24},6, selected ? Color{0.86f,0.92f,1.0f,1} : Color{0.98f,0.99f,1.0f,1});
            RoundOutline({22, y, 286, 24},6, selected ? Color{0.36f,0.58f,0.88f,1} : Color{0.80f,0.85f,0.92f,1});
            const Color material_color = MaterialColor(layer.material, layer.color);
            RoundRect({25, y + 7, 10, 10}, 3, material_color);
            const std::string material_label = layer.material == "custom" ? "CUSTOM N/K" : layer.material;
            Text(39, y + 5, material_label, 1.02f, {0.14f,0.20f,0.28f,1});
            RoundRect({120, y + 1, 62, 22},5, app->active_geom_thickness_row == i ? Color{0.95f,0.98f,1.0f,1} : Color{1.0f,1.0f,1.0f,0.60f});
            RoundOutline({120, y + 1, 62, 22},5, app->active_geom_thickness_row == i ? Color{0.20f,0.52f,0.82f,1} : Color{0.74f,0.80f,0.88f,1}, app->active_geom_thickness_row == i ? 1.8f : 1.0f);
            const std::string thick = app->active_geom_thickness_row == static_cast<int>(i) ? app->edit_buffer : Format(layer.thickness_nm, 2);
            Text(124, y + 5, thick, 1.02f, {0.14f,0.20f,0.28f,1});
            RoundRect({186, y + 1, 88, 22},5,{1.0f,1.0f,1.0f,0.55f});
            RoundOutline({186, y + 1, 88, 22},5,{0.74f,0.80f,0.88f,1});
            Text(190, y + 5, layer.geometry_mode == "uniform" ? "UNIFORM" : "NON-UNIFORM",
                 0.90f, {0.24f,0.38f,0.54f,1});
            DrawButton({280, y, 12, 11}, "+",false,false,1.0f);
            DrawButton({280, y + 13, 12, 11}, "-",false,false,1.0f);
        }
        if (app->material_menu_open)
        {
            std::vector<std::string> names{"custom"};
            for (const MaterialRecord& record : SharedMaterialDatabase().materials())
                names.push_back(record.name);
            constexpr int visible_items = 12;
            const float menu_y = 158.0f;
            RoundRect({22, 132, 286, 318}, 8, {0.98f,0.99f,1.0f,1});
            RoundOutline({22, 132, 286, 318}, 8, {0.56f,0.70f,0.88f,1});
            Text(32, 140, "MATERIAL LIBRARY  /  SCROLL TO BROWSE", 0.92f,
                 {0.28f,0.42f,0.58f,1});
            for (int slot = 0; slot < visible_items; ++slot)
            {
                const int index = app->material_menu_scroll + slot;
                if (index >= static_cast<int>(names.size()))
                    break;
                const Rect item{22, menu_y + static_cast<float>(slot) * 24.0f, 286, 22};
                const std::array<float, 3> fallback{0.44f,0.68f,0.92f};
                const Color material_color = MaterialColor(names[static_cast<size_t>(index)], fallback);
                if (slot % 2)
                    Quad(item, {0.94f,0.97f,1.0f,0.72f});
                RoundRect({28, item.y + 6, 10, 10}, 3, material_color);
                const std::string display_name = names[static_cast<size_t>(index)] == "custom"
                    ? "CUSTOM (MANUAL N/K FROM JSON)" : names[static_cast<size_t>(index)];
                Text(44, item.y + 5, display_name, 0.98f,
                     {0.16f,0.22f,0.30f,1});
            }
            const std::string range = std::to_string(app->material_menu_scroll + 1) + "-" +
                std::to_string(std::min(app->material_menu_scroll + visible_items,
                                        static_cast<int>(names.size()))) + " / " +
                std::to_string(names.size());
            Text(224, 434, range, 0.82f, {0.42f,0.54f,0.66f,1});
        }
    }
    else if (app->active_tab == kSolverTab)
    {
        DrawToggle({22,130,286,66},"Numeric Precision","Single","Double",
                   app->config.precision == "double");
        DrawToggle({22,230,286,66},"Matrix Approximation","Order 12","Order 18",
                   app->config.approximation_order == 18);
        Text(22,322,"SETTINGS APPLY TO THE NEXT RUN",1.0f,{0.27f,0.49f,0.58f,1});
    }

    DrawButton({22,height - 68.0f,286,44}, app->running ? "FORCE STOP SIMULATION" : "RUN SIMULATION", true, false);
    DrawButton({22,height - 122.0f,89,36}, "LOAD JSON");
    DrawButton({119,height - 122.0f,89,36}, "SAVE JSON");
    Text(22,height - 144.0f, "DEFAULT_SIMULATION.JSON", 0.92f, {0.42f,0.54f,0.66f,1});

    Text(image_far.x + 16, image_far.y + 12, "Far-field Image", 1.05f, {0.44f,0.74f,0.82f,1});
    Text(image_near.x + 16, image_near.y + 12, "Near-field Image", 1.05f, {0.44f,0.74f,0.82f,1});
    if (app->result.focus_count > 0)
    {
        DrawButton({image_far.x+image_far.w-172,image_far.y+7,24,22},"<",false,false,1.15f);
        const double focus = app->result.focus_nm[static_cast<size_t>(app->selected_focus)];
        const std::string focus_text = Format(focus,2)+" NM";
        Text(image_far.x+image_far.w-142,image_far.y+12,focus_text,0.90f,{0.26f,0.42f,0.60f,1});
        DrawButton({image_far.x+image_far.w-76,image_far.y+7,24,22},">",false,false,1.15f);
    }
    DrawButton({image_far.x+image_far.w-48,image_far.y+7,42,22},"SAVE",false,false,0.85f);
    DrawButton({image_near.x+image_near.w-48,image_near.y+7,42,22},"SAVE",false,false,0.85f);
    DrawImagePanel(*app, {image_far.x, image_far.y + 34, image_far.w, image_far.h - 40}, app->farfield_texture,
                   "NO FAR-FIELD RESULT", "RUN THE SIMULATION");
    DrawImagePanel(*app, {image_near.x, image_near.y + 34, image_near.w, image_near.h - 40}, app->nearfield_texture,
                   "NO NEAR-FIELD RESULT", "RUN THE SIMULATION");
    RoundOutline(structure,14,{0.62f,0.74f,0.88f,0.92f},1.2f);
    RoundOutline(image_far,14,{0.62f,0.74f,0.88f,0.92f},1.2f);
    RoundOutline(image_near,14,{0.62f,0.74f,0.88f,0.92f},1.2f);
    RoundRect({structure.x+14,structure.y+14,114,25},12,{0.89f,0.95f,1.0f,0.96f});
    RoundRect({structure.x+24,structure.y+23,7,7},4,{0.19f,0.92f,0.68f,1});
    Text(structure.x+38,structure.y+21,"Live 3D View",1.0f,{0.28f,0.42f,0.58f,1});

    const Rect status_box{image_far.x, height - 42.0f, image_far.w, 28.0f};
    RoundRect(status_box, 10, {0.92f,0.96f,1.0f,0.98f});
    RoundOutline(status_box, 10, {0.68f,0.78f,0.90f,1});
    RoundRect({status_box.x+10,status_box.y+10,8,8},4,
              app->running ? Color{1,0.50f,0.20f,1} : Color{0.22f,0.94f,0.62f,1});
    Text(status_box.x+26,status_box.y+7,"STATUS",0.92f,{0.34f,0.50f,0.64f,1});
    Text(status_box.x+82,status_box.y+6,app->status,1.05f,
         app->result.error.empty() ? Color{0.18f,0.38f,0.26f,1} : Color{0.80f,0.24f,0.18f,1});
    const std::string render_info = "RENDER  " + app->render_gpu;
    const std::string compute_info = "COMPUTE  " + (app->compute_gpu.empty() ? "N/A" : app->compute_gpu);
    Text(left + 22, height - 32, render_info, 0.95f, {0.38f,0.50f,0.62f,1});
    Text(left + 22, height - 16, compute_info, 0.95f, {0.46f,0.58f,0.70f,1});
}
} // namespace

int main(int argc, char** argv)
{
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow(1440, 900, "Simulator GPU Optical Workbench", nullptr, nullptr);
    if (!window)
    {
        std::fprintf(stderr, "Failed to create OpenGL window. Check DISPLAY/Wayland configuration.\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    App app;
    app.window = window;
    app.executable_dir = std::filesystem::absolute(argv[0]).parent_path().string();
    app.json_path = (std::filesystem::path(app.executable_dir) /
                     "resources/config/default_simulation.json").string();
    app.orientation = Multiply(AxisAngle({1,0,0}, 38.0f * kPi / 180.0f),
                               AxisAngle({0,0,1}, -28.0f * kPi / 180.0f));
    if (argc > 1)
        app.json_path = argv[1];
    std::string load_error;
    if (LoadSimulationConfig(app.json_path, &app.config, &load_error))
        app.status = "JSON CONFIG LOADED";
    else
        app.status = "DEFAULT CONFIG - " + load_error;
    std::string material_error;
    if (!LoadSharedMaterialDatabase(
            (std::filesystem::path(app.executable_dir) / "resources/tztek_materials.csv").string(),
            &material_error))
        app.status = material_error;
    g_app = &app;

    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    if (renderer)
    {
        std::string r(renderer);
        if (auto pos = r.find('('); pos != std::string::npos)
            r.resize(pos);
        while (!r.empty() && r.back() == ' ')
            r.pop_back();
        app.render_gpu = r;
    }
    app.compute_gpu = QueryNvidiaGpuName();

    const std::filesystem::path executable_dir(app.executable_dir);
    const std::filesystem::path font_candidates[] = {
        executable_dir / "resources/fonts/LiberationSans-Bold.ttf",
        executable_dir.parent_path() / "resources/fonts/LiberationSans-Bold.ttf",
        std::filesystem::path("resources/fonts/LiberationSans-Bold.ttf"),
    };
    bool font_ready = false;
    for (const std::filesystem::path& font_path : font_candidates)
        if (g_font.Initialize(font_path.string().c_str()))
        {
            font_ready = true;
            break;
        }
    if (!font_ready)
    {
        std::fprintf(stderr, "Failed to load bundled font resources/fonts/LiberationSans-Bold.ttf.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwSetKeyCallback(window, KeyCallback);
    glfwSetCharCallback(window, CharCallback);
    glfwSetScrollCallback(window, ScrollCallback);

    while (!glfwWindowShouldClose(window))
    {
        const auto frame_time = std::chrono::steady_clock::now();
        const float dt = std::clamp(std::chrono::duration<float>(frame_time - app.last_frame).count(), 0.0f, 0.05f);
        app.last_frame = frame_time;
        glfwPollEvents();
        glfwGetCursorPos(window, &app.mouse_x, &app.mouse_y);
        const bool down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if (down && !app.mouse_pressed)
            HandleClick(&app, width, height);
        if (down && app.dragging_view)
        {
            const Quat delta = TrackballDelta(app.structure_panel,
                                              app.last_mouse_x, app.last_mouse_y,
                                              app.mouse_x, app.mouse_y);
            app.orientation = Multiply(delta, app.orientation);
            app.spin = delta;
        }
        else if (!down)
        {
            const float frame_factor = std::clamp(dt * 60.0f, 0.0f, 3.0f);
            app.orientation = Multiply(SlerpIdentity(app.spin, frame_factor), app.orientation);
            app.spin = SlerpIdentity(app.spin, std::exp(-4.2f * dt));
        }
        if (!down)
            app.dragging_view = false;
        app.mouse_pressed = down;
        app.last_mouse_x = app.mouse_x;
        app.last_mouse_y = app.mouse_y;

        if (app.running && app.worker_pid > 0)
        {
            int worker_status = 0;
            const pid_t done = waitpid(app.worker_pid, &worker_status, WNOHANG);
            if (done == app.worker_pid)
            {
                app.worker_pid = -1;
                app.running = false;
                if (ReadSimulationResult(app.worker_result_path, &app.result))
                {
                    if (app.result.error.empty())
                    {
                        app.status = "SIMULATION COMPLETE";
                        app.selected_focus = 0;
                        UploadResultTexture(&app);
                    }
                    else
                        app.status = app.result.error;
                }
                else
                    app.status = "CUDA WORKER EXITED WITHOUT A RESULT";
            }
        }

        DrawUi(&app, width, height);
        glfwSwapBuffers(window);
    }

    if (app.running)
        StopSimulation(&app);
    if (!app.worker_config_path.empty())
        std::filesystem::remove(app.worker_config_path);
    if (!app.worker_result_path.empty())
        std::filesystem::remove(app.worker_result_path);
    if (app.farfield_texture)
        glDeleteTextures(1, &app.farfield_texture);
    if (app.nearfield_texture)
        glDeleteTextures(1, &app.nearfield_texture);
    if (g_font.texture)
        glDeleteTextures(1, &g_font.texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
