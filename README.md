# Parameters =

# Numerics

**ImageCalculationMode** (str) 'vector', 'scalar'
**ImageCalculationMethod** (str) 'abbe', 'hopkins'
**Hopkins_SettingType** (str) 'order' 'Threshold'
**Hopkins_Order** (int) Truncation order of SOCS method
**Hopkins_Threshold** (float) Between 0 and 1, default 0.95

# Projection

**Aberration_Zernike** (Tensor) Coefficients of Zernike polynomials
**Magnification** (float) Magnification factor
**NA** (float) Numerical aperture of object
**IndexImage** (float) Refractive index of image
**FocusRange** (Tensor) Focus range

# Mask

**Period_X** (float) X-period in nanometers
**Period_Y** (float) Y-period in nanometers

# Source

**Wavelength** (float) Wavelength in nanometers
**PntNum** (int) Discretized number
**Shape** (str) 'annular', 'multipole', 'dipolecirc', 'quadrupole', 'pixel', 'quasar'
**Irradiance** (float)
**PolarizationType** 'x_pol', 'y_pol', 'c_pol', 'd_pol', 'r_pol', 't_pol', 'line_pol', 'custom'
**PolarizationParameters.CoherencyMatrix** (tensor) 2-by-2 Hermitian matrix
**SigmaIn** (float) for all shapes except multipole and pixel
**SigmaOut** (float) for all shapes except multipole and pixel
**SigmaCenter** (float) for multipole shape
**SigmaRadius** (float) for multipole shape
**OpenAngle** (float) for quasar and dipolecirc shapes
**RotateAngle** (float) for dipolecirc and multipole shapes
**PoleNumber** (int) for multipole shape
**Source_Mask** (tensor)
**PSFEnable** (bool) source blur
**PSFSigma** (float) source blur sigma
**SPointX** (tensor) for pixel shape
**SPointY** (tensor) for pixel shape
**SPointValue** (tensor) for pixel shape