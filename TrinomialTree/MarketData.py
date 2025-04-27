from datetime import date

class MarketData:
    """ Classes des paramètres du marché """
    def __init__(self, S0 : float, r : float, vol : float, div : float, div_date : date): # Prix spot, taux d'intérêt, volatilité, dividende, date de dividende
        # Attributs de la classe :
        self.price: float = S0
        self.rf: float = r
        self.vol: float = vol
        self.div: float = div
        self.div_date: date = div_date