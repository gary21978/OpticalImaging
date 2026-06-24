#include "simulation_backend.h"
#include "simulation_config.h"

#include <cstdio>
#include <filesystem>

int main(int argc, char** argv)
{
    std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s config.json result.bin\n", argv[0]);
        return 2;
    }
    SimulationConfig config;
    std::string error;
    if (!LoadSimulationConfig(argv[1], &config, &error))
    {
        SimulationResult result;
        result.error = error;
        WriteSimulationResult(argv[2], result);
        return 3;
    }
    const SimulationResult result = RunSimulation(config);
    if (!WriteSimulationResult(argv[2], result))
        return 4;
    return result.error.empty() ? 0 : 5;
}
