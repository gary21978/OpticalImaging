from Simulok import Simulok
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulok params")
    # 运行设备
    parser.add_argument("--device", type=str, default='cuda',
                        help="optional, default 'cuda'")
    # RCWA参数
    parser.add_argument("--pngpath", type=str, default='./SimPattern/DSW/LL_180_230nm.png',
                        help="png path, default './Geo/LL_180_230nm.png'")
    # order 期望两个整数，顺序为 ox oy（在原代码中对顺序有说明）
    parser.add_argument("--order", type=int, nargs=2, default=[7, 7],
                        help="two integers: [ox, oy], default [15, 15]")
    parser.add_argument("--lamda", type=int, default=365,
                        help="wavelength in nm, default 365")
    # period 期望一个或两个整数，若为一个则 x 与 y 相同
    parser.add_argument("--period", type=int, nargs='+', default=[9000],
                        help="period in nm, one value or two values [Lx Ly], default [9000],当前暂时不支持不同值")
    parser.add_argument("--resolution", type=int, nargs='+', default=[256, 256],
                        help="render resolution, default [256, 256]")
    
    # 放大倍率
    parser.add_argument("--magnification", type=float, default=100.0,
                        help="magnification，default 100.0")
    
    # 最终保存可视化结果的像素尺寸
    parser.add_argument("--pixelsize", type=int, default=90,
                        help="define pixelsize in nm, default 90")

    # 光源参数
    # 形状类型
    parser.add_argument("--shape", type=str, default='annular',
                        help="define source shape, options: 'annular', 'dipolecirc', 'quasar', 'rect' ；default 'annular'")
    # 偏振
    parser.add_argument("--polarization", type=str, default='None',
                        help="define polarization, e.g. x_pol, y_pol ；default 'None' ")
    # 形状参数
    parser.add_argument("--sigmaOut", type=float, default=0.9,
                        help="SigmaOut for annular Source, default 0.9")
    parser.add_argument("--sigmaIn", type=float, default=0.15,
                        help="SigmaIn for annular Source, default 0.15")
    parser.add_argument("--openAngle", type=float, default=0.0,
                        help="Open angle for dipolecirc / quasar Source, default 0.0")
    parser.add_argument("--rotateAngle", type=float, default=0.0,
                        help="Rotate angle for dipolecirc / quasar / rect Source, default 0.0")
    parser.add_argument("--RectX", type=float, default=0.0,
                        help="width for Rect Source, default 0.0")
    parser.add_argument("--RectY", type=float, default=0.0,
                        help="height for Rect Source, default 0.0")
    
    # 成像光瞳参数
    # 形状类型
    parser.add_argument("--imaging_pupil_shape", type=str, default='annular',
                        help="define imaging pupil shape, options: 'annular', 'dipolecirc', 'quasar', 'rect' ；default 'annular'")
    # 形状参数
    parser.add_argument("--imaging_pupil_sigmaOut", type=float, default=0.9,
                        help="SigmaOut for annular imaging pupil, default 0.9")
    parser.add_argument("--imaging_pupil_sigmaIn", type=float, default=0.15,
                        help="SigmaIn for annular imaging pupil, default 0.15")
    parser.add_argument("--imaging_pupil_openAngle", type=float, default=0.0,
                        help="Open angle for dipolecirc / quasar imaging pupil, default 0.0")
    parser.add_argument("--imaging_pupil_rotateAngle", type=float, default=0.0,
                        help="Rotate angle for dipolecirc / quasar / rect imaging pupil, default 0.0")  
    parser.add_argument("--imaging_pupil_RectX", type=float, default=0.0,
                        help="width for Rect imaging pupil, default 0.0")
    parser.add_argument("--imaging_pupil_RectY", type=float, default=0.0,
                        help="height for Rect imaging pupil, default 0.0")
    # 偏振
    parser.add_argument("--imaging_pupil_polarization", type=str, default='None',
                        help="未生效！！！define imaging pupil polarization, e.g. x_pol, y_pol ；default 'None' ")
    
    
    args = parser.parse_args()
    
    pipeline = Simulok(args)
    pipeline.Run()    # 基本仿真

    # pipeline.save_name = "OPEN_BRIDGE_Case2_NA"
    ## 遍历参数
    # pipeline.RunIterative()

    ## 宽光谱仿真
    # pipeline.band = 'ILine'
    # pipeline.RunBroadband()