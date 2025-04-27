import pandas as pd
import numpy as np
from VasicekModel import VasicekModel

# ----------------------------------------------------------------------
# Classe de pricing d'une obligation à coupons utilisant VasicekModel
# ----------------------------------------------------------------------
class CouponBondPricer:
    """
    Pricing d'une obligation à coupons via une instance de VasicekModel.
    """
    def __init__(self, vasicek_model: VasicekModel, r0: float, maturity: float, coupon_rate: float, face: float = 100.0, freq: int = 2):
        self.model = vasicek_model
        self.r0 = r0
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.face = face
        self.freq = freq


    def price(self) -> float:
        """
        Calcule le prix d'une obligation à coupon :
        - r0          : taux court initial
        - maturity    : maturité en années
        - coupon_rate : taux de coupon annuel (ex. 0.03 pour 3%)
        """
        n_periods = int(self.maturity * self.freq)
        c_period = self.coupon_rate / self.freq * self.face
        price = 0.0
        # Actualisation des coupons
        for i in range(1, n_periods + 1):
            t_i = i / self.freq
            P0ti = self.model.price(self.r0, t_i)
            price += c_period * P0ti
        # Actualisation du nominal
        price += self.face * self.model.price(self.r0, self.maturity)
        return price

    def macaulay_duration(self) -> float:
        """
        Calcule la duration de Macaulay (en années).
        """
        n_periods = int(self.maturity * self.freq)
        c_period = self.coupon_rate / self.freq * self.face
        price = self.price()
        weighted_sum = sum(
            (i / self.freq) * c_period * self.model.price(self.r0, i / self.freq)
            for i in range(1, n_periods + 1)
        )
        # ajout du nominal à maturité
        weighted_sum += self.maturity * self.face * self.model.price(self.r0, self.maturity)
        return weighted_sum / price

    def convexity(self) -> float:
        """
        Calcule la convexité (en années^2).
        """
        n_periods = int(self.maturity * self.freq)
        c_period = self.coupon_rate / self.freq * self.face
        price = self.price()
        weighted2_sum = sum(
            (i / self.freq)**2 * c_period * self.model.price(self.r0, i / self.freq)
            for i in range(1, n_periods + 1)
        )
        # ajout du nominal à T
        weighted2_sum += self.maturity**2 * self.face * self.model.price(self.r0, self.maturity)
        return weighted2_sum / price

# ----------------------------------------------------------------------
# Exemple d'utilisation
# ----------------------------------------------------------------------
if __name__ == '__main__':

    # Chargement des données (Taux court : Euribor 3mois)
    df = pd.read_csv('data\data_vasicek.csv', sep=';', parse_dates=['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df['Rate'] = df['Dernier'].str.replace(',', '.').astype(float) / 100
    rates = df['Rate'].values
    delta_days = (df['Date'].iloc[1] - df['Date'].iloc[0]).days
    dt = delta_days / 52  # donnée hebdomadaire

    # VasicekModel pour pricer les ZC
    vasicek = VasicekModel()
    vasicek.calibrate(rates, dt, margin=0.00)   # calibration

    r0 = rates[-1]
    maturity = 5.0              # maturité 5 ans
    coupon = 0.025              # 2.5% annuel

    # Pricing ZC et obligation à coupons
    zc_price = vasicek.price(r0, maturity)
    coupon_pricer = CouponBondPricer(vasicek, r0, maturity, coupon, face=100.0, freq=2)
    bond_price = coupon_pricer.price()
    dur = coupon_pricer.macaulay_duration()
    conv = coupon_pricer.convexity()
    
    # Affichage des paramètres calibrés
    print("Parametres calibres :")
    print(f"  a     = {vasicek.a:.6f}  (vitesse de rappel)")
    print(f"  k     = {vasicek.k:.6f}  (taux moyen long terme)")
    print(f"  sigma = {vasicek.sigma:.6f}  (volatilite)")
    
    # Prix
    print(f"Prix ZC ({maturity} ans)    : {zc_price:.6f}")
    print(f"Prix obligation      : {bond_price:.6f}")

    # Métriques pour l'obligation avec coupons
    print(f"Duration de Macaulay : {dur:.4f} ans")
    print(f"Convexite            : {conv:.4f} annees^2")

