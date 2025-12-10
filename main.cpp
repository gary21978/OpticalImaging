#include <iostream>
#include "RCWA.h"

int main()
{
    try
    {
        // Initialize RCWA
        RCWA rcwa(1.0e6, {5, 5}, {1.0e-6, 1.0e-6}, RCWA::DType::COMPLEX64);

        // Configure layers
        rcwa.addInputLayer(1.0);
        rcwa.addOutputLayer(1.0);
        rcwa.addLayer(0.5e-6); // Homogeneous layer

        // Set incident angle (45 degrees)
        rcwa.setIncidentAngle(M_PI / 4, 0.0, "input");

        // Configure source
        rcwa.sourcePlanewave({1.0, 0.0}, "forward");

        // Solve S-matrix
        rcwa.solveGlobalSmatrix();

        // Calculate fields (simplified)
        float x_axis[] = {0.0, 0.1e-6, 0.2e-6};
        float y_axis[] = {0.0, 0.1e-6, 0.2e-6};
        auto [Ex, Ey, Ez] = rcwa.fieldXY(x_axis, y_axis, 3, 3);

        std::cout << "RCWA simulation completed successfully!" << std::endl;

        // Cleanup
        rcwa.freeDeviceTensor(Ex);
        rcwa.freeDeviceTensor(Ey);
        rcwa.freeDeviceTensor(Ez);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}