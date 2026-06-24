#include "material_database.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace
{
MaterialDatabase g_database;

std::vector<std::string> ParseCsvRow(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i)
    {
        const char c = line[i];
        if (c == '"')
        {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"')
            {
                field.push_back('"');
                ++i;
            }
            else
                quoted = !quoted;
        }
        else if (c == ',' && !quoted)
        {
            fields.push_back(field);
            field.clear();
        }
        else
            field.push_back(c);
    }
    fields.push_back(field);
    return fields;
}
}

bool MaterialDatabase::Load(const std::string& path, std::string* error)
{
    std::ifstream input(path);
    if (!input)
    {
        if (error) *error = "Cannot open material cache: " + path;
        return false;
    }
    materials_.clear();
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        const std::vector<std::string> fields = ParseCsvRow(line);
        if (fields.size() != 5)
            continue;
        const std::string& name = fields[0];
        const std::string& source = fields[1];
        double wavelength = 0.0;
        double n = 0.0;
        double k = 0.0;
        try
        {
            wavelength = std::stod(fields[2]);
            n = std::stod(fields[3]);
            k = std::stod(fields[4]);
        }
        catch (const std::exception&)
        {
            continue;
        }
        auto it = std::find_if(materials_.begin(), materials_.end(),
                               [&](const MaterialRecord& record) { return record.name == name; });
        if (it == materials_.end())
        {
            materials_.push_back({name, source, {}});
            it = materials_.end() - 1;
        }
        it->samples.push_back({wavelength, n, k});
    }
    for (MaterialRecord& record : materials_)
        std::sort(record.samples.begin(), record.samples.end(),
                  [](const MaterialSample& a, const MaterialSample& b) {
                      return a.wavelength_nm < b.wavelength_nm;
                  });
    if (materials_.empty())
    {
        if (error) *error = "Material cache is empty.";
        return false;
    }
    return true;
}

bool MaterialDatabase::Lookup(const std::string& name, double wavelength_nm, double* n, double* k) const
{
    const auto material = std::find_if(materials_.begin(), materials_.end(),
                                       [&](const MaterialRecord& record) { return record.name == name; });
    if (material == materials_.end() || material->samples.empty())
        return false;
    const auto upper = std::lower_bound(material->samples.begin(), material->samples.end(), wavelength_nm,
                                        [](const MaterialSample& sample, double wavelength) {
                                            return sample.wavelength_nm < wavelength;
                                        });
    if (upper == material->samples.begin() || upper == material->samples.end())
        return false;
    const MaterialSample& b = *upper;
    const MaterialSample& a = *(upper - 1);
    const double t = (wavelength_nm - a.wavelength_nm) / (b.wavelength_nm - a.wavelength_nm);
    if (n) *n = a.n + t * (b.n - a.n);
    if (k) *k = a.k + t * (b.k - a.k);
    return true;
}

const MaterialDatabase& SharedMaterialDatabase()
{
    return g_database;
}

bool LoadSharedMaterialDatabase(const std::string& path, std::string* error)
{
    return g_database.Load(path, error);
}
