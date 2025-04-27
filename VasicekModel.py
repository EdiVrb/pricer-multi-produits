import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Classe de pricing Zero-Coupon selon Vasicek (avec calibration)
# ----------------------------------------------------------------------
class VasicekModel:
    """
    Modèle de Vasicek : calibration par OLS + pricing d'une ZC.
    """
    def __init__(self):
        self.a = None
        self.k = None
        self.sigma = None

    def calibrate(self, rates: np.ndarray, dt: float, margin: float = 0.0):
        """
        Calibration des paramètres (a, k, sigma) par régression OLS.
        - rates : série historique des taux courts
        - dt    : pas de temps en années
        - margin: décalage à ajouter aux taux pour la stabilité
        """
        rates_arr = np.asarray(rates, dtype=float) + margin
        r_t = rates_arr[:-1]
        r_next = rates_arr[1:]
        X = np.vstack([r_t, np.ones_like(r_t)]).T
        theta, *_ = np.linalg.lstsq(X, r_next, rcond=None)
        alpha, beta = theta
        self.a = (1 - alpha) / dt
        self.k = beta / (self.a * dt)
        eps = r_next - (alpha * r_t + beta)
        self.sigma = np.std(eps, ddof=1) / np.sqrt(dt)

    def price(self, r0: float, maturity: float) -> float:
        """
        Prix d'une zero-coupon de maturité `maturity` à partir du taux initial r0.
        """
        if None in (self.a, self.k, self.sigma):
            raise ValueError("Modèle non calibré. Appeler d'abord `calibrate`.")
        a, k, sigma = self.a, self.k, self.sigma
        T = maturity
        # Instantaneous volatility of ZC
        sigma_b = sigma * (1 - np.exp(-a * T)) / a
        # A(T) calculation
        A = np.exp((sigma_b / sigma - T) * (k - sigma**2 / (2 * a**2)) - sigma_b**2 / (4 * a))
        # ZC price
        price_zc = A * np.exp(- (sigma_b / sigma) * r0)
        return price_zc
