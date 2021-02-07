

class Distribution:
    def __init__(self, name, f_density, f_data_generator):
        self.name = name
        self.f_density = f_density
        self.f_data_generator = f_data_generator
