import math
from scipy.stats import norm

class OptBS:
    """
    Classe permettant de calculer le prix par la méthode de B&S
    """

    def __init__(self, start_date, mat_date, start_price: float, rate: float, vol: float,
                 opt_type: str, exercise: str, k: float):
        """calcul des composantes de la formule dans le constructeur"""
        time_to_mat: float = (mat_date - start_date).days / 365
        self.d1: float = ((math.log(start_price / k) + (rate + (vol ** 2) / 2) * time_to_mat) /
                          (vol * math.sqrt(time_to_mat)))
        self.d2: float = self.d1 - vol * math.sqrt(time_to_mat)
        # Computation du prix par l'appel de la méthode bs_price
        self.price_bs: float = self.bs_price(start_price, k, rate, time_to_mat, self.d1, self.d2, opt_type)

    def bs_price(self, start_price: float, k: float, rate: float,
                 time_to_mat: float, d1: float, d2: float, opt_type: str) -> float:
        """Méthode de Calcul du prix selon que l'option est un Call ou un Put"""
        if opt_type == "Call" :
            return start_price * norm.cdf(d1, 0, 1) - k * math.exp(-rate * time_to_mat) * norm.cdf(d2, 0, 1,)
        # Calcul du prix de B&S pour un put européen
        else:
            return k * math.exp(-rate * time_to_mat) * norm.cdf(-d2, 0, 1) - start_price * norm.cdf(-d1, 0, 1)