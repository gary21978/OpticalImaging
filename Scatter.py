class Scatter:
    def __init__(self):
        self.Order = 0
        self.Period_X = 1000  # nanometer
        self.Period_Y = 1000  # nanometer
        
        self.ScatterResult = {}
        self.FloquetMode = []

        self.resolution_x = 128
        self.resolution_y = 128
        
        
    def GetSpectrum(self, z_posi):
        h = self.ScatterResult['z_posi'][z_posi]['Floquet_mode']
        return h
    
