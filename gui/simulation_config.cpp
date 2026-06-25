#include "simulation_config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace
{
std::string ReadAll(const std::string& path)
{
    std::ifstream input(path);
    return input ? std::string((std::istreambuf_iterator<char>(input)), {}) : std::string();
}

size_t SkipSpace(const std::string& text, size_t pos)
{
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
        ++pos;
    return pos;
}

bool FindValue(const std::string& text, const std::string& key, size_t* value_pos)
{
    const std::string needle = "\"" + key + "\"";
    size_t pos = text.find(needle);
    if (pos == std::string::npos)
        return false;
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos)
        return false;
    *value_pos = SkipSpace(text, pos + 1);
    return true;
}

bool ReadNumber(const std::string& text, const std::string& key, double* value)
{
    size_t pos = 0;
    if (!FindValue(text, key, &pos))
        return false;
    char* end = nullptr;
    *value = std::strtod(text.c_str() + pos, &end);
    return end != text.c_str() + pos && std::isfinite(*value);
}

bool ReadInt(const std::string& text, const std::string& key, int* value)
{
    double parsed = 0.0;
    if (!ReadNumber(text, key, &parsed))
        return false;
    *value = static_cast<int>(std::lround(parsed));
    return true;
}

bool ReadBool(const std::string& text, const std::string& key, bool* value)
{
    size_t pos = 0;
    if (!FindValue(text, key, &pos))
        return false;
    if (text.compare(pos, 4, "true") == 0)
    {
        *value = true;
        return true;
    }
    if (text.compare(pos, 5, "false") == 0)
    {
        *value = false;
        return true;
    }
    return false;
}

bool ReadString(const std::string& text, const std::string& key, std::string* value)
{
    size_t pos = 0;
    if (!FindValue(text, key, &pos) || pos >= text.size() || text[pos] != '"')
        return false;
    ++pos;
    std::string out;
    while (pos < text.size() && text[pos] != '"')
    {
        if (text[pos] == '\\' && pos + 1 < text.size())
            ++pos;
        out.push_back(text[pos++]);
    }
    if (pos >= text.size())
        return false;
    *value = out;
    return true;
}

bool ReadColor(const std::string& text, std::array<float, 3>* color)
{
    size_t pos = 0;
    if (!FindValue(text, "color", &pos) || pos >= text.size() || text[pos] != '[')
        return false;
    for (int i = 0; i < 3; ++i)
    {
        pos = SkipSpace(text, pos + 1);
        char* end = nullptr;
        const double value = std::strtod(text.c_str() + pos, &end);
        if (end == text.c_str() + pos)
            return false;
        (*color)[i] = static_cast<float>(std::clamp(value, 0.0, 1.0));
        pos = static_cast<size_t>(end - text.c_str());
        pos = text.find(i == 2 ? ']' : ',', pos);
        if (pos == std::string::npos)
            return false;
    }
    return true;
}

bool ReadNumberArray(const std::string& text, const std::string& key, std::vector<double>* values)
{
    size_t pos = 0;
    if (!values || !FindValue(text, key, &pos) || pos >= text.size() || text[pos] != '[')
        return false;
    std::vector<double> parsed;
    ++pos;
    while (pos < text.size())
    {
        pos = SkipSpace(text, pos);
        if (pos < text.size() && text[pos] == ']')
            break;
        char* end = nullptr;
        const double value = std::strtod(text.c_str() + pos, &end);
        if (end == text.c_str() + pos)
            return false;
        parsed.push_back(value);
        pos = SkipSpace(text, static_cast<size_t>(end - text.c_str()));
        if (pos < text.size() && text[pos] == ',')
            ++pos;
        else if (pos >= text.size() || text[pos] != ']')
            return false;
    }
    if (!parsed.empty())
        *values = std::move(parsed);
    return !parsed.empty();
}

std::vector<std::string> ReadLayerObjects(const std::string& text)
{
    size_t pos = 0;
    if (!FindValue(text, "layers", &pos) || pos >= text.size() || text[pos] != '[')
        return {};
    std::vector<std::string> objects;
    int array_depth = 1;
    for (++pos; pos < text.size() && array_depth > 0; ++pos)
    {
        if (text[pos] == ']')
            --array_depth;
        if (text[pos] != '{')
            continue;
        const size_t begin = pos;
        int depth = 1;
        bool quoted = false;
        for (++pos; pos < text.size() && depth > 0; ++pos)
        {
            if (text[pos] == '"' && (pos == 0 || text[pos - 1] != '\\'))
                quoted = !quoted;
            if (!quoted && text[pos] == '{')
                ++depth;
            if (!quoted && text[pos] == '}')
                --depth;
        }
        if (depth == 0)
            objects.push_back(text.substr(begin, pos - begin));
    }
    return objects;
}

std::vector<std::string> ReadObjectArray(const std::string& text, const std::string& key)
{
    size_t pos = 0;
    if (!FindValue(text, key, &pos) || pos >= text.size() || text[pos] != '[')
        return {};
    std::vector<std::string> objects;
    int array_depth = 1;
    for (++pos; pos < text.size() && array_depth > 0; ++pos)
    {
        if (text[pos] == ']')
            --array_depth;
        if (text[pos] != '{')
            continue;
        const size_t begin = pos;
        int depth = 1;
        bool quoted = false;
        for (++pos; pos < text.size() && depth > 0; ++pos)
        {
            if (text[pos] == '"' && (pos == 0 || text[pos - 1] != '\\'))
                quoted = !quoted;
            if (!quoted && text[pos] == '{')
                ++depth;
            if (!quoted && text[pos] == '}')
                --depth;
        }
        if (depth == 0)
            objects.push_back(text.substr(begin, pos - begin));
    }
    return objects;
}
} // namespace

SimulationConfig DefaultSimulationConfig()
{
    SimulationConfig config;
    config.image_width = 1024;
    config.image_height = 512;
    config.layers = {
        {"Pattern", "SiO2", 100.0, 1.4745, 0.0, true, "png", "resources/geometry/DSW.png", {0.10f, 0.78f, 0.98f}, {}},
        {"TiN absorber", "TiN", 21.0, 2.0669, 1.0563, false, "uniform", "", {0.92f, 0.33f, 0.18f}, {}},
        {"Nitride spacer", "Si3N4", 100.0, 1.3800, 0.0, false, "uniform", "", {0.43f, 0.36f, 0.92f}, {}},
    };
    return config;
}

void UpdateNyquistGrid(SimulationConfig* config)
{
    if (!config)
        return;
    auto grid_for_order = [](int order) {
        const int required = std::max(16, 4 * std::max(order, 0) + 2);
        int value = 1;
        while (value < required)
            value <<= 1;
        return value;
    };
    config->geometry_x = grid_for_order(config->order_x);
    config->geometry_y = grid_for_order(config->order_y);
    config->image_width = std::max(1024, config->image_width);
    const double aspect = config->period_x_nm > 0.0 ? config->period_y_nm / config->period_x_nm : 1.0;
    config->image_height = std::max(8, static_cast<int>(std::lround(config->image_width * std::max(0.1, aspect))));
}

bool ValidateSimulationConfig(const SimulationConfig& c, std::string* error)
{
    auto fail = [error](const char* message) {
        if (error)
            *error = message;
        return false;
    };
    if (c.wavelength_nm <= 0.0 || c.period_x_nm <= 0.0 || c.period_y_nm <= 0.0)
        return fail("Wavelength and periods must be positive.");
    if (c.order_x < 0 || c.order_y < 0 || c.order_x > 64 || c.order_y > 64)
        return fail("RCWA orders must be between 0 and 64.");
    SimulationConfig nyquist = c;
    UpdateNyquistGrid(&nyquist);
    if (c.geometry_x < nyquist.geometry_x || c.geometry_y < nyquist.geometry_y)
        return fail("Geometry grid violates the Nyquist sampling requirement.");
    if (c.image_width < 8 || c.image_height < 8)
        return fail("Image grid must be at least 8 x 8.");
    if (c.sigma_inner < 0.0 || c.sigma_outer <= c.sigma_inner || c.sigma_outer > 1.0)
        return fail("Sigma must satisfy 0 <= inner < outer <= 1.");
    if (c.layers.empty())
        return fail("At least one layer is required.");
    if (c.approximation_order != 12 && c.approximation_order != 18)
        return fail("Matrix approximation order must be 12 or 18.");
    if ((c.source_shape == "dipole" || c.source_shape == "quasar") &&
        c.source_open_angle_deg <= 0.0)
        return fail("Dipole and quasar sources require an open angle greater than 0 degrees.");
    if (c.precision != "single" && c.precision != "double")
        return fail("Precision must be single or double.");
    if (c.zernike.size() != 37)
        return fail("Exactly 37 Zernike coefficients are required.");
    for (const LayerConfig& layer : c.layers)
    {
        if (layer.thickness_nm <= 0.0)
            return fail("Every layer thickness must be positive.");
        for (const ShapeConfig& shape : layer.shapes)
            if ((shape.type != "rect" && shape.type != "circle") ||
                shape.size_x <= 0.0 || shape.size_y <= 0.0)
                return fail("Layer shapes must be rect/circle with positive size.");
        if (layer.geometry_mode != "uniform" && layer.geometry_mode != "png")
            return fail("Layer geometry mode must be uniform or png.");
    }
    if (error)
        error->clear();
    return true;
}

bool LoadSimulationConfig(const std::string& path, SimulationConfig* config, std::string* error)
{
    if (!config)
        return false;
    const std::string text = ReadAll(path);
    if (text.empty())
    {
        if (error)
            *error = "Cannot read JSON file: " + path;
        return false;
    }

    SimulationConfig parsed = DefaultSimulationConfig();
    ReadNumber(text, "period_x_nm", &parsed.period_x_nm);
    ReadNumber(text, "period_y_nm", &parsed.period_y_nm);
    ReadNumber(text, "wavelength_nm", &parsed.wavelength_nm);
    ReadInt(text, "order_x", &parsed.order_x);
    ReadInt(text, "order_y", &parsed.order_y);
    ReadInt(text, "geometry_x", &parsed.geometry_x);
    ReadInt(text, "geometry_y", &parsed.geometry_y);
    ReadInt(text, "image_width", &parsed.image_width);
    ReadInt(text, "image_height", &parsed.image_height);
    ReadNumber(text, "sigma_outer", &parsed.sigma_outer);
    ReadNumber(text, "sigma_inner", &parsed.sigma_inner);
    ReadString(text, "source_shape", &parsed.source_shape);
    ReadNumber(text, "source_open_angle_deg", &parsed.source_open_angle_deg);
    ReadNumber(text, "source_rotate_angle_deg", &parsed.source_rotate_angle_deg);
    ReadNumber(text, "source_rect_x", &parsed.source_rect_x);
    ReadNumber(text, "source_rect_y", &parsed.source_rect_y);
    ReadString(text, "polarization", &parsed.polarization);
    ReadNumber(text, "polarization_angle_deg", &parsed.polarization_angle_deg);
    ReadInt(text, "approximation_order", &parsed.approximation_order);
    ReadString(text, "precision", &parsed.precision);
    ReadNumber(text, "incident_theta_deg", &parsed.incident_theta_deg);
    ReadNumber(text, "incident_phi_deg", &parsed.incident_phi_deg);
    ReadNumber(text, "input_n_real", &parsed.input_n_real);
    ReadNumber(text, "input_n_imag", &parsed.input_n_imag);
    ReadNumber(text, "substrate_n_real", &parsed.substrate_n_real);
    ReadNumber(text, "substrate_n_imag", &parsed.substrate_n_imag);
    ReadString(text, "substrate_material", &parsed.substrate_material);
    if (!ReadNumber(text, "focus_nm", &parsed.focus_nm))
    {
        std::vector<double> legacy_focus;
        if (ReadNumberArray(text, "focus_nm", &legacy_focus) && !legacy_focus.empty())
            parsed.focus_nm = legacy_focus.front();
    }
    size_t zpos = 0;
    if (FindValue(text, "zernike", &zpos) && zpos < text.size() && text[zpos] == '[')
    {
        parsed.zernike.assign(37, 0.0);
        for (int i = 0; i < 37; ++i)
        {
            zpos = SkipSpace(text, zpos + 1);
            char* end = nullptr;
            parsed.zernike[i] = std::strtod(text.c_str() + zpos, &end);
            if (end == text.c_str() + zpos)
                break;
            zpos = text.find(i == 36 ? ']' : ',', static_cast<size_t>(end - text.c_str()));
            if (zpos == std::string::npos)
                break;
        }
    }

    const std::vector<std::string> objects = ReadLayerObjects(text);
    if (!objects.empty())
    {
        parsed.layers.clear();
        for (const std::string& object : objects)
        {
            LayerConfig layer;
            ReadString(object, "name", &layer.name);
            ReadString(object, "material", &layer.material);
            ReadNumber(object, "thickness_nm", &layer.thickness_nm);
            ReadNumber(object, "n_real", &layer.n_real);
            ReadNumber(object, "n_imag", &layer.n_imag);
            ReadBool(object, "patterned", &layer.patterned);
            ReadString(object, "geometry_mode", &layer.geometry_mode);
            ReadString(object, "png_path", &layer.png_path);
            ReadColor(object, &layer.color);
            const std::vector<std::string> shape_objects = ReadObjectArray(object, "shapes");
            for (const std::string& shape_object : shape_objects)
            {
                ShapeConfig shape;
                ReadString(shape_object, "type", &shape.type);
                ReadNumber(shape_object, "center_x", &shape.center_x);
                ReadNumber(shape_object, "center_y", &shape.center_y);
                ReadNumber(shape_object, "size_x", &shape.size_x);
                ReadNumber(shape_object, "size_y", &shape.size_y);
                layer.shapes.push_back(shape);
            }
            parsed.layers.push_back(layer);
        }
    }
    UpdateNyquistGrid(&parsed);
    if (!ValidateSimulationConfig(parsed, error))
        return false;
    *config = std::move(parsed);
    return true;
}

bool SaveSimulationConfig(const std::string& path, const SimulationConfig& c, std::string* error)
{
    if (!ValidateSimulationConfig(c, error))
        return false;
    std::ofstream out(path);
    if (!out)
    {
        if (error)
            *error = "Cannot write JSON file: " + path;
        return false;
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"period_x_nm\": " << c.period_x_nm << ",\n"
        << "  \"period_y_nm\": " << c.period_y_nm << ",\n"
        << "  \"wavelength_nm\": " << c.wavelength_nm << ",\n"
        << "  \"order_x\": " << c.order_x << ",\n"
        << "  \"order_y\": " << c.order_y << ",\n"
        << "  \"geometry_x\": " << c.geometry_x << ",\n"
        << "  \"geometry_y\": " << c.geometry_y << ",\n"
        << "  \"image_width\": " << c.image_width << ",\n"
        << "  \"image_height\": " << c.image_height << ",\n"
        << "  \"sigma_outer\": " << c.sigma_outer << ",\n"
        << "  \"sigma_inner\": " << c.sigma_inner << ",\n"
        << "  \"source_shape\": \"" << c.source_shape << "\",\n"
        << "  \"source_open_angle_deg\": " << c.source_open_angle_deg << ",\n"
        << "  \"source_rotate_angle_deg\": " << c.source_rotate_angle_deg << ",\n"
        << "  \"source_rect_x\": " << c.source_rect_x << ",\n"
        << "  \"source_rect_y\": " << c.source_rect_y << ",\n"
        << "  \"polarization\": \"" << c.polarization << "\",\n"
        << "  \"polarization_angle_deg\": " << c.polarization_angle_deg << ",\n"
        << "  \"approximation_order\": " << c.approximation_order << ",\n"
        << "  \"precision\": \"" << c.precision << "\",\n"
        << "  \"incident_theta_deg\": " << c.incident_theta_deg << ",\n"
        << "  \"incident_phi_deg\": " << c.incident_phi_deg << ",\n"
        << "  \"input_n_real\": " << c.input_n_real << ",\n"
        << "  \"input_n_imag\": " << c.input_n_imag << ",\n"
        << "  \"substrate_n_real\": " << c.substrate_n_real << ",\n"
        << "  \"substrate_n_imag\": " << c.substrate_n_imag << ",\n"
        << "  \"substrate_material\": \"" << c.substrate_material << "\",\n"
        << "  \"focus_nm\": " << c.focus_nm << ",\n"
        << "  \"zernike\": [";
    for (size_t i = 0; i < c.zernike.size(); ++i)
        out << (i ? ", " : "") << c.zernike[i];
    out << "],\n"
        << "  \"layers\": [\n";
    for (size_t i = 0; i < c.layers.size(); ++i)
    {
        const LayerConfig& l = c.layers[i];
        out << "    {\"name\": \"" << l.name << "\", \"material\": \"" << l.material
            << "\", \"thickness_nm\": " << l.thickness_nm
            << ", \"n_real\": " << l.n_real << ", \"n_imag\": " << l.n_imag
            << ", \"patterned\": " << (l.patterned ? "true" : "false")
            << ", \"geometry_mode\": \"" << l.geometry_mode << "\""
            << ", \"png_path\": \"" << l.png_path << "\""
            << ", \"color\": [" << l.color[0] << ", " << l.color[1] << ", " << l.color[2] << "]"
            << ", \"shapes\": [";
        for (size_t j = 0; j < l.shapes.size(); ++j)
        {
            const ShapeConfig& s = l.shapes[j];
            out << (j ? ", " : "")
                << "{\"type\": \"" << s.type << "\", \"center_x\": " << s.center_x
                << ", \"center_y\": " << s.center_y << ", \"size_x\": " << s.size_x
                << ", \"size_y\": " << s.size_y << "}";
        }
        out << "]}";
        out << (i + 1 == c.layers.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return static_cast<bool>(out);
}
