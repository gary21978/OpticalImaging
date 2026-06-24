#pragma once

#include <string>
#include <vector>

struct MaterialSample
{
    double wavelength_nm = 0.0;
    double n = 1.0;
    double k = 0.0;
};

struct MaterialRecord
{
    std::string name;
    std::string source;
    std::vector<MaterialSample> samples;
};

class MaterialDatabase
{
  public:
    bool Load(const std::string& path, std::string* error = nullptr);
    bool Lookup(const std::string& name, double wavelength_nm, double* n, double* k) const;
    const std::vector<MaterialRecord>& materials() const { return materials_; }

  private:
    std::vector<MaterialRecord> materials_;
};

const MaterialDatabase& SharedMaterialDatabase();
bool LoadSharedMaterialDatabase(const std::string& path, std::string* error = nullptr);
