from datetime import date

class TimeParam:
    """ Classe permettant de récupérer les dates de début et de maturité du contrat, ainsi que le nombre de pas """
    def __init__(self, start_date: date, mat_date: date, n_step: int):
        self.start_date: date = start_date
        self.mat_date: date = mat_date
        self.n_step: int = n_step