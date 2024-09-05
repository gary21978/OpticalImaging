class Numerics:
    # Imaging model => 'vector'  'scalar'
    ImageCalculationMode: str = 'scalar'
    # Calculation model =>'abbe' 'hopkins'
    ImageCalculationMethod: str = 'hopkins'
    # Truncation method for hopkins model => 'order' 'Threshold'
    Hopkins_SettingType: str = 'order'
    Hopkins_Order: int = 100
    Hopkins_Threshold: float = 0.95  # (0, 1)
    ScatterGrid_X: int = 100
    ScatterGrid_Y: int = 100
    ScatterOrder_X: int = 15
    ScatterOrder_Y: int = 15

    def __init__(self):
        pass