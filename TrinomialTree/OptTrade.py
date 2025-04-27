from datetime import date

class OptTrade2:
    """Classe permettant de récupérer les paramètres associé au type du contrat """
    def __init__(self, opt_type: str, opt_exercice: str, k: float):
        self.opt_type: str = opt_type
        self.opt_exe: str = opt_exercice
        self.K: float = k

    def VI(self, St: float) -> float:
        if self.opt_type == "Call":
            return max(St - self.K, 0)

        elif self.opt_type == "Put":
            return max(self.K - St, 0)