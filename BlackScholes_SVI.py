import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BSSVIModel:
    """
    Classe qui calcule les prix d'options européennes (call et put)
    et permet la calibration du modèle SVI avec calcul de la volatilité implicite et du taux implicite.
    Inclut aussi le calcul des grecs et l'affichage du smile.
    """
    def __init__(self, S, T, K, market_prices, r, option_type='call'):
        self.S = S
        self.T = T
        self.K = np.array(K)
        self.k = np.log(self.K / S)
        self.market_prices = np.array(market_prices)
        self.r = r
        self.option_type = option_type
        self.params = None

    @staticmethod
    def bs_price(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def bs_greeks(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        return delta, gamma, vega, theta, rho

    @staticmethod
    def svi_volatility(k, a, b, rho, m, sigma, T):
        w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        return np.sqrt(w / T)

    @staticmethod
    def implied_rate(S, K, T, q, call_price, put_price):
        return - (1 / T) * np.log((put_price - call_price + S * np.exp(-q * T)) / K)

    def svi_price_objective(self, params):
        a, b, rho, m, sigma = params
        if b <= 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return 1e10
        svi_vols = self.svi_volatility(self.k, a, b, rho, m, sigma, self.T)
        model_prices = np.array([
            self.bs_price(self.S, k_, self.T, self.r, vol, self.option_type)
            for k_, vol in zip(self.K, svi_vols)
        ])
        return np.sum((model_prices - self.market_prices) ** 2)

    def calibrate(self):
        initial_guess = [0.1, 0.1, 0.0, 0.0, 0.1]
        bounds = [(-1, 2), (1e-6, 10), (-0.999, 0.999), (-5, 5), (1e-6, 5)]
        result = minimize(
            self.svi_price_objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )
        if result.success:
            self.params = result.x
            return result.x
        else:
            raise ValueError("Échec de la calibration SVI : " + result.message)

    def get_vols_and_prices(self):
        if self.params is None:
            raise ValueError("SVI non calibré. Appelez d'abord .calibrate()")
        a, b, rho, m, sigma = self.params
        svi_vols = self.svi_volatility(self.k, a, b, rho, m, sigma, self.T)
        svi_prices = [
            self.bs_price(self.S, k_, self.T, self.r, v, self.option_type)
            for k_, v in zip(self.K, svi_vols)
        ]
        return svi_vols, svi_prices

    def plot_smile(self):
        if self.params is None:
            raise ValueError("SVI non calibré. Appelez d'abord .calibrate()")
        a, b, rho, m, sigma = self.params
        vol_svi = self.svi_volatility(self.k, a, b, rho, m, sigma, self.T)
        plt.figure(figsize=(8, 5))
        plt.plot(self.K, vol_svi, label='Smile SVI', lw=2, color='blue')
        plt.scatter(self.K, vol_svi, color='red', label='Vols implicites SVI')
        plt.xlabel('Strike')
        plt.ylabel('Volatilité implicite')
        plt.title('Smile de volatilité implicite (modèle SVI)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        return plt
    
# === Utilisation avec fichier Excel ===
if __name__ == "__main__":
    # Chargement des données depuis Excel
    df = pd.read_excel("data\options_data.xlsx")
    S = 100
    T = 0.25
    q = 0.0

    # Extraction des colonnes du fichier
    market_call_prices = df['Call_Price'].values
    market_put_prices = df['Put_Price'].values
    strikes = df['Strike'].values

    # Calcul du taux implicite pour chaque strike, puis moyenne
    taux_implicites = [BSSVIModel.implied_rate(S, K, T, q, c, p)
                       for K, c, p in zip(strikes, market_call_prices, market_put_prices)]
    r_moyen = np.mean(taux_implicites)

    # Initialisation et calibration du modèle SVI
    model = BSSVIModel(S, T, strikes, market_call_prices, r_moyen)
    params = model.calibrate()
    svi_vols, svi_prices = model.get_vols_and_prices()

    # Affichage des paramètres calibrés
    # print("\nParamètres SVI calibrés :")
    # print(f"a={params[0]:.4f}, b={params[1]:.4f}, rho={params[2]:.4f}, m={params[3]:.4f}, sigma={params[4]:.4f}\n")
    # Affichage des résultats pour chaque strike
    # for k, p_mkt, p_svi, v in zip(strikes, market_call_prices, svi_prices, svi_vols):
    #     print(f"Strike {k}: Prix marché = {p_mkt:.4f}, Prix SVI = {p_svi:.4f}, Vol SVI = {v:.2%}")

    # Exemple de calcul de volatilité pour un strike donné (105 ici)
    K = 105  # exemple de strike
    vol_call = model.svi_volatility(np.log(K / S), params[0],params[1],params[2],params[3],params[4],0.25)
    price_call = model.bs_price(S, K, T, r_moyen, vol_call, option_type='call')
    print(price_call)

    