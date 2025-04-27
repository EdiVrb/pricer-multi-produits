import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class NSModel:
    """
    Classe pour calibration et utilisation de la courbe Nelson–Siegel.
    """
    def __init__(self, curve_file: str, sheet_name: str = 'RateCurve'):
        self.curve_file = curve_file
        self.sheet_name = sheet_name
        self.beta0 = None
        self.beta1 = None
        self.beta2 = None
        self.lam   = None

    def _convert_mat(self, pillar: str) -> float:
        p = pillar.strip().upper()
        if p.endswith("M"): return int(p[:-1]) / 12
        if p.endswith("Y"): return int(p[:-1])
        raise ValueError(f"Format inconnu: {pillar}")

    def calibrate(self):
        """Lit le fichier Excel et calibre β0, β1, β2, λ."""
        df = pd.read_excel(self.curve_file,
                           sheet_name=self.sheet_name,
                           engine='openpyxl')
        df['mat'] = df['Pillar'].map(self._convert_mat)
        x = df['mat'].values.astype(float)
        y = df['Rate'].values.astype(float)
        popt, _ = curve_fit(self._nelson_siegel, x, y)
        self.beta0, self.beta1, self.beta2, self.lam = popt

    @staticmethod
    def _nelson_siegel(t, b0, b1, b2, lam):
        """
        Formule NS vectorisée pour t scalaire ou array.
        Renvoie rendement en pourcentage.
        """
        t_arr = np.asarray(t, dtype=float)
        # préventivement ignorer warnings de divisions par zéro
        with np.errstate(divide='ignore', invalid='ignore'):
            # term1 = (1 - exp(-t/lam)) / (t/lam)
            term1 = (1.0 - np.exp(-t_arr / lam)) / (t_arr / lam)
            # term2 = term1 - exp(-t/lam)
            term2 = term1 - np.exp(-t_arr / lam)
        # remplacer les NaN / inf créés par 0<=t_arr<epsilon par leurs limites
        term1 = np.where(np.isclose(t_arr, 0.0), 1.0, term1)
        term2 = np.where(np.isclose(t_arr, 0.0), 0.0, term2)
        return b0 + b1 * term1 + b2 * term2

    def spot(self, t: float) -> float:
        """Taux spot en fraction (0.025 pour 2.5%)."""
        y_pc = self._nelson_siegel(t,
                                   self.beta0,
                                   self.beta1,
                                   self.beta2,
                                   self.lam)
        return y_pc / 100.0

    def forward(self, t0: float, t1: float) -> float:
        """Calcul du taux forward de [t0, t1]."""
        y0 = self.spot(t0)
        y1 = self.spot(t1)
        return (y1 * t1 - y0 * t0) / (t1 - t0)

if __name__ == '__main__':
    ns = NSModel('data\RateCurve.xlsx')
    ns.calibrate()
    print("Paramètres NS:", ns.beta0, ns.beta1, ns.beta2, ns.lam)
    print("Spot 0:", ns.spot(0.0))
    print("Spot 2Y:", ns.spot(2.0))
    print("Forward 1Y->2Y:", ns.forward(1.0, 2.0))
