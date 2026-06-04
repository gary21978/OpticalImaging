import math
import torch
import torch.nn.functional as F
import sys

def conv2(matr, ker, mode='same'):
    matr_x, matr_y = matr.size()
    ker_x, ker_y = ker.size()
    matr_4d = matr.reshape((1, 1, matr_x, matr_y))
    ker_4d = ker.reshape((1, 1, ker_x, ker_y))
    conv = F.conv2d(matr_4d, ker_4d, padding=mode)
    conv = conv.reshape((matr_x, matr_y))
    return conv

class Source:
    def __init__(self):
        self.PntNum = 101
        self.Wavelength = 365
        self.Shape = "annular"
        # Source shape :
        # 'annular''dipolecirc''quasar''rect'

        self.SigmaOut = 0.9
        self.SigmaIn = 0.0

        self.OpenAngle = math.pi/3
        self.RotateAngle = 0
        self.RectX = 0.5
        self.RectY = 0.5
        self.PoleNumber = 2
        self.Source_Mask = None  # tensor
        self.SigmaCenter = 0.5
        self.SigmaRadius = 0.1

        # parameter for pixel source, maybe 2d tensor
        self.SPointX = torch.zeros(self.PntNum)
        self.SPointY = torch.zeros(self.PntNum)
        self.SPointValue = torch.zeros(self.PntNum)
        # PSF parameter
        # source blur parameter initialization
        self.PSFEnable = False
        self.PSFSigma = 0.02
        # source polarization
        self.PolarizationType = 'x_pol'
        self.PolarizationParameters = PolarizationParameters()
        self.source_data = SourceData(self.PntNum, self.PntNum)
        
    def Cal_valid(self, kx, ky):
        # 确保输入参数为 PyTorch 张量
        if not isinstance(kx, torch.Tensor):
            kx = torch.tensor(kx, dtype=torch.float32)
        if not isinstance(ky, torch.Tensor):
            ky = torch.tensor(ky, dtype=torch.float32)
            
        if ((self.Shape).lower() == "pixel"):
            # pixel source
            raise ValueError("unsupported illumination, on developing......")
            
        elif ((self.Shape).lower() == "annular"):
            r2 = kx**2 + ky**2
            return (r2 <= self.SigmaOut**2) & (r2 >= self.SigmaIn**2)

        elif ((self.Shape).lower() == "dipolecirc"):
            open_angle = self.OpenAngle
            rotate_angle = self.RotateAngle
    
            if not (-math.pi / 2 <= rotate_angle <= math.pi / 2):
                raise ValueError("rotateAngle must be in [-pi/2, pi/2]")
            if not (0 <= open_angle <= math.pi / 2):
                raise ValueError("openAngle must be in [0, pi/2]")
    
            radius = torch.sqrt(kx**2 + ky**2)
            theta = torch.atan2(ky, kx)
    
            # 方向约束：主瓣在 rotate_angle 方向，张角为 open_angle
            cos_diff = torch.cos(theta - rotate_angle)
            angle_ok = torch.abs(cos_diff) >= math.cos(open_angle / 2)
    
            return (radius <= self.SigmaOut) & (radius >= self.SigmaIn) & angle_ok
        
        elif ((self.Shape).lower() == "rect"):
            rect_x = self.RectX
            rect_y = self.RectY
            rotate_angle = self.RotateAngle
    
            if rect_x**2 + rect_y**2 > 1.0:
                raise ValueError("RectX^2 + RectY^2 must be <= 1 for normalized pupil")
    
            # 逆旋转坐标系
            cos_t = math.cos(-rotate_angle)
            sin_t = math.sin(-rotate_angle)
            x_prime = kx * cos_t - ky * sin_t
            y_prime = kx * sin_t + ky * cos_t
    
            return (torch.abs(x_prime) <= rect_x) & (torch.abs(y_prime) <= rect_y)
            
        elif ((self.Shape).lower() == "quasar"):
            open_angle = self.OpenAngle
            rotate_angle = self.RotateAngle
    
            if not (-math.pi / 2 <= rotate_angle <= math.pi / 2):
                raise ValueError("rotateAngle must be in [-pi/2, pi/2]")
            if not (0 <= open_angle <= math.pi / 2):
                raise ValueError("openAngle must be in [0, pi/2]")
    
            radius = torch.sqrt(kx**2 + ky**2)
            theta = torch.atan2(ky, kx)
    
            # 四极：0°, 90°, 180°, 270° 附近四个瓣
            def in_sector(center):
                diff = torch.remainder(theta - center + math.pi, 2 * math.pi) - math.pi  # wrap to [-pi, pi]
                return torch.abs(diff) <= open_angle / 2
    
            sector0 = in_sector(rotate_angle)
            sector90 = in_sector(rotate_angle + math.pi / 2)
            sector180 = in_sector(rotate_angle + math.pi)
            sector270 = in_sector(rotate_angle - math.pi / 2)
    
            angle_ok = sector0 | sector90 | sector180 | sector270
    
            return (radius <= self.SigmaOut) & (radius >= self.SigmaIn) & angle_ok

        elif ((self.Shape).lower() == "multipole"):
            
            # pixel source
            raise ValueError("unsupported illumination, on developing......")

        else:
            raise ValueError("unsupported illumination")

    
    def Calc_SourceSimple(self):
        self.source_data = self.Calc_SourceAll()
        Low_Weight = self.source_data.Value < 1e-5

        self.source_data.X = self.source_data.X[~Low_Weight]
        self.source_data.Y = self.source_data.Y[~Low_Weight]
        self.source_data.Value = self.source_data.Value[~Low_Weight]
        return self.source_data

    def Calc_SourceValid(self):
        data = self.Calc_SourceAll()
        self.source_data.Value = torch.where(
                (data.X.pow(2) + data.Y.pow(2)) > 1,
                0, data.Value
            )
        return self.source_data

    def Calc_SourceAll(self):
        Source_Coordinate_X = torch.linspace(-1, 1, self.PntNum)
        if ((self.Shape).lower() == "pixel"):
            # pixel source
            # Checking the mesh size of the source for pixel sources
            # to ensure that it conforms to the defined specifications.
            # FIXME: The coordinates and values of the pupil 
            #        need to be defined and the data format checked
            if (self.SPointX.numel() != self.PntNum**2):
                raise ValueError(
                    'Source Matrix Size X and difined PntNum are not matched '
                    )

            if (self.SPointY.numel() != self.PntNum**2):
                raise ValueError(
                    'Source Matrix Size Y and difined PntNum are not matched '
                    )

            if (self.SPointValue.numel() != self.PntNum**2):
                raise ValueError(
                    'Source Matrix Size Value and difined \
                    PntNum are not matched '
                    )
            self.source_data.X = self.SPointX
            self.source_data.Y = self.SPointY
            self.source_data.Value = self.SPointValue

        elif ((self.Shape).lower() == "annular"):
            self.source_data = CalculateAnnularSourceMatrix(
                self.SigmaOut, self.SigmaIn,
                Source_Coordinate_X, Source_Coordinate_X
            )

        elif ((self.Shape).lower() == "quasar"):
            openAngle = self.OpenAngle
            rotateAngle = self.RotateAngle
            if (rotateAngle > math.pi/2) | (rotateAngle < -1 * math.pi / 2):
                raise ValueError(
                    'error: roate angle must be in the range of [-pi/2,pi/2] '
                    )
            if (openAngle <= 0) | (openAngle >= math.pi/2):
                raise ValueError(
                    'error: open angle must be in the range of [0,pi/2] '
                    )

            self.source_data = CalculateQuasarSourceMatrix(
                self.SigmaOut, self.SigmaIn, openAngle,
                Source_Coordinate_X, Source_Coordinate_X
                )


        elif ((self.Shape).lower() == "dipolecirc"):
            openAngle = self.OpenAngle
            rotateAngle = self.RotateAngle
            if (rotateAngle > math.pi/2) | (rotateAngle < -1 * math.pi / 2):
                raise ValueError(
                    'error: roate angle must be in the range of [-pi/2,pi/2] '
                    )
            if (openAngle < 0) | (openAngle > math.pi/2):
                raise ValueError(
                    'error: open angle must be in the range of [0,pi/2] '
                    )
            self.source_data = CalculateDipoleSourceMatrix(
                self.SigmaOut, self.SigmaIn, openAngle, rotateAngle,
                Source_Coordinate_X, Source_Coordinate_X
                )
        
        elif ((self.Shape).lower() == "rect"):
            if self.RectX**2 + self.RectY**2 > 1:
                raise ValueError(
                    'error: RectX**2 + RectY**2  must be less than 1 '
                    )
                
            self.source_data = CalculateRectSourceMatrix(
                self.RectX, self.RectY, self.RotateAngle,
                Source_Coordinate_X, Source_Coordinate_X
                )
    
        elif ((self.Shape).lower() == "multipole"):
            rotateAngle = self.RotateAngle
            if (rotateAngle > math.pi/2) | (rotateAngle < -1 * math.pi / 2):
                raise ValueError(
                    'error: roate angle must be in the range of [-pi/2,pi/2] '
                    )
            self.source_data = CalculateMultiCircSourceMatrix(
                self.SigmaCenter, self.SigmaRadius, self.PoleNumber,
                rotateAngle, Source_Coordinate_X, Source_Coordinate_X
                )

        else:
            raise ValueError("unsupported illumination")

        sizeX, sizeY = self.source_data.Value.size()

        if (self.Source_Mask is not None):
            self.source_data.Value = torch.mul(
                self.source_data.Value, self.Source_Mask
                )
        # Data serialization and add source blur
        if (sizeX == sizeY):
            # Add source blur
            if self.PSFEnable:
                kernelSize = round(self.PntNum/10)*2 + 1
                kernelEdge = 1 / (self.PntNum - 1) * (kernelSize - 1)
                kernelX, kernelY = torch.meshgrid(
                    torch.linspace(
                        -kernelEdge, kernelEdge, kernelSize
                        ),
                    torch.linspace(
                        -kernelEdge, kernelEdge, kernelSize
                        ),
                    indexing='ij'
                    )
                kernel = 1 / math.sqrt(2 * math.pi) / self.PSFSigma  \
                    * torch.exp(
                        - (kernelX.pow(2)+kernelY.pow(2)) /
                        self.PSFSigma ** 2
                        )
                kernel = kernel[~torch.all(kernel < 1e-6, 0)]
                kernel_tr = torch.transpose(kernel, 0, 1)
                kernel = kernel_tr[~torch.all(kernel_tr < 1e-6, 1)]
                kernel = kernel/torch.sum(kernel)
                self.source_data.Value = conv2(
                    self.source_data.Value, kernel, mode='same'
                    )
            # Set center point to 0
            self.source_data.Value[int((sizeX-1)/2), int((sizeY-1)/2)] = 0
            self.source_data = ConvertSourceMatrix2SourceData(self.source_data)
        return self.source_data

    # source polarization
    def Calc_PolarizationMap(self, theta, rho):
        if (self.PolarizationType == 'x_pol'):
            PolarizedX = torch.ones(theta.size())
            PolarizedY = torch.zeros(theta.size())
        elif (self.PolarizationType == 'y_pol'):
            PolarizedX = torch.zeros(theta.size())
            PolarizedY = torch.ones(theta.size())
        elif (self.PolarizationType == 'c_pol'):
            PolarizedX = math.sqrt(.5) * torch.ones(theta.size())
            PolarizedY = math.sqrt(.5) * 1j * torch.ones(theta.size())
        elif (self.PolarizationType == 'd_pol'):
            PolarizedX = math.sqrt(.5) * torch.ones(theta.size())
            PolarizedY = math.sqrt(.5) * torch.ones(theta.size())    
        elif (self.PolarizationType == 'r_pol'):
            PolarizedX = torch.cos(theta)
            PolarizedY = torch.sin(theta)
        elif (self.PolarizationType == 't_pol'):
            PolarizedX = torch.sin(theta)
            PolarizedY = -1 * torch.cos(theta)
        elif (self.PolarizationType == 'line_pol'):
            PolarizedX = torch.mul(
                torch.sin(self.PolarizationParameters.Angle),
                torch.ones(theta.size())
                )
            PolarizedY = torch.mul(
                torch.cos(self.PolarizationParameters.Angle),
                torch.ones(theta.size())
                )
        else:
            raise ValueError("unsupported polarization type")

        biz = rho < sys.float_info.epsilon
        if (len(biz) > sys.float_info.epsilon):
            PolarizedX[biz] = 0
            PolarizedY[biz] = 0
        PolarizedX = PolarizedX.to(torch.complex64)
        PolarizedY = PolarizedY.to(torch.complex64)
        return PolarizedX, PolarizedY


def CalculateAnnularSourceMatrix(
    SigmaOut, SigmaIn,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y, indexing='ij')
    Radius = torch.sqrt(s.X.pow(2) + s.Y.pow(2))
    Index = (Radius <= SigmaOut) & (Radius >= SigmaIn)
    s.Value[Index] = 1.0/((SigmaOut**2 - SigmaIn**2) * math.pi)
    return s

def CalculateQuasarSourceMatrix(
    SigmaOut, SigmaIn, openAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y, indexing='ij')
    Radius = torch.sqrt(s.X.pow(2) + s.Y.pow(2))
    theta = torch.atan2(s.Y, s.X)
    Indextheta1 = (torch.abs(theta) <= (openAngle / 2)) | \
        ((torch.abs(theta) >= (math.pi - openAngle / 2)) &
         (torch.abs(theta) <= math.pi))
    Indextheta2 = (torch.abs(theta - math.pi / 2) <= openAngle / 2) | \
        (torch.abs(theta + math.pi / 2) <= openAngle / 2)
    Indextheta = Indextheta1 | Indextheta2
    Index = (Radius <= SigmaOut) & (Radius >= SigmaIn) & Indextheta
    s.Value[Index] = 0.5/((SigmaOut**2 - SigmaIn**2) * openAngle)
    return s

def CalculateRectSourceMatrix(
    RectX, RectY, rotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y, indexing='ij')
    
    # 逆旋转点坐标
    cos_theta = torch.cos(torch.tensor(-rotateAngle))
    sin_theta = torch.sin(torch.tensor(-rotateAngle))
    x_prime = s.X * cos_theta - s.Y * sin_theta
    y_prime = s.X * sin_theta + s.Y * cos_theta
    
    # 检查点是否在矩形内
    Index = (torch.abs(x_prime) <= RectX) & (torch.abs(y_prime) <= RectY)
    
    # 设置矩形内的值，归一化基于矩形面积
    s.Value[Index] = 1.0 / (4 * RectX * RectY)
    
    return s

def CalculateDipoleSourceMatrix(
    SigmaOut, SigmaIn, openAngle, rotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y, indexing='ij')
    Radius = torch.sqrt(s.X.pow(2) + s.Y.pow(2))
    theta = torch.atan2(s.Y, s.X)
    Indextheta = (torch.abs(torch.cos(theta - rotateAngle)) >=
                  math.cos(openAngle / 2))
    Index = (Radius <= SigmaOut) & (Radius >= SigmaIn) & Indextheta
    s.Value[Index] = 1.0/((SigmaOut**2 - SigmaIn**2) * openAngle)
    return s

# multipule source
def CalculateMultiCircSourceMatrix(
    SigmaCenter, SigmaRadius, PoleNumber, RotateAngle,
    Source_Coordinate_X, Source_Coordinate_Y
):
    s = SourceData(len(Source_Coordinate_X), len(Source_Coordinate_Y))
    s.X, s.Y = torch.meshgrid(Source_Coordinate_X, Source_Coordinate_Y, indexing='ij')
    rotateStep = 2 * math.pi / PoleNumber
    for i in range(PoleNumber):
        xCenter = SigmaCenter * math.cos(RotateAngle + i * rotateStep)
        yCenter = SigmaCenter * math.sin(RotateAngle + i * rotateStep)
        Radius2 = (s.X - xCenter).pow(2) + (s.Y - yCenter).pow(2)
        Index = (Radius2 <= SigmaRadius**2)
        s.Value[Index] = 1.0/PoleNumber/(SigmaRadius**2)
    return s

def ConvertSourceMatrix2SourceData(s):
    rSquare = s.X ** 2 + s.Y ** 2
    sizeSourceX = s.X.shape[0]
    s.Value[rSquare > 1] = 0

    SourceData.X = torch.reshape(s.X, (sizeSourceX * sizeSourceX, 1))
    SourceData.Y = torch.reshape(s.Y, (sizeSourceX * sizeSourceX, 1))
    SourceData.Value = torch.reshape(s.Value, (sizeSourceX * sizeSourceX, 1))

    return SourceData

class SourceData:
    def __init__(self, x, y):
        self.X = torch.zeros((x, y))
        self.Y = torch.zeros((x, y))
        self.Value = torch.zeros((x, y))

class PolarizationParameters:
    def __init__(self):
        self.Degree = 1  # Partially polarized light is not yet supported
        self.Angle = torch.tensor(0)  # Polarization direction: start at positive half of x-axis, counterclockwise