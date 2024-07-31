class Numerics:
    # Imaging model => 'vector'  'scalar'
    ImageCalculationMode: str = 'vector'
    # Calculation model =>'abbe' 'hopkins'
    ImageCalculationMethod: str = 'abbe'
    # Truncation method for hopkins model => 'order' 'Threshold'
    Hopkins_SettingType: str = 'order'
    Hopkins_Order: int = 50
    Hopkins_Threshold: float = 0.95  # (0, 1)
    def __init__(self):
        pass