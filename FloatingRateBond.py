import pandas as pd

from NelsonSiegel import NSModel
from VasicekModel import VasicekModel

class FloatingRateBondPricer:
    """
    Classe pour pricer une obligation à taux flottant :
      - NSModel pour calculer les forwards
      - VasicekModel pour actualiser via zéros-coupons
    """
    def __init__(self,
                 curve_file: str,
                 hist_file: str,
                 sheet_curve: str = 'RateCurve'):
        # Initialisation des modèles
        self.ns        = NSModel(curve_file, sheet_curve)
        self.vas       = VasicekModel()
        self.hist_file = hist_file

    def calibrate(self):
        """Calibrage NS et Vasicek."""
        # 1) Calibrer la courbe Nelson–Siegel
        self.ns.calibrate()
        # 2) Charger et calibrer Vasicek
        df = pd.read_csv(self.hist_file,
                         sep=';',
                         parse_dates=['Date'],
                         dayfirst=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Rate'] = df['Dernier'].str.replace(',', '.').astype(float) / 100.0
        rates = df['Rate'].values
        dt    = df['Date'].diff().dt.days.iloc[1] / 360.0
        self.vas.calibrate(rates, dt)
        self.vas.r0 = rates[-1]

    def forward_rate(self, t0: float, t1: float) -> float:
        """
        Renvoie le taux forward de la période [t0, t1] calculé
        sur la courbe Nelson-Siegel.
        """
        return self.ns.forward(t0, t1)

    def price(self,
              maturity: float,
              spread: float,
              notional: float,
              frequency: int) -> float:
        """
        Calcule le prix de la FRN :
          PV = sum_{i=1..n} [ (forward + spread) * var_t * N * P_vas(0,t_i) ]
               + N * P_vas(0, maturity)
        """
        var_t   = 1.0 / frequency
        n   = int(maturity * frequency)
        r0  = self.vas.r0
        pv  = 0.0
        # Boucle sur chaque période
        for i in range(1, n+1):
            t0  = (i-1) * var_t
            t1  = i      * var_t
            fwd = self.forward_rate(t0, t1)
            cf  = (fwd + spread) * var_t * notional
            pv += cf * self.vas.price(r0, t1)
        # Actualisation du principal
        pv += notional * self.vas.price(r0, maturity)
        return pv
    

    # ---- MÉTRIQUES DE RISQUE ----
    def macaulay_duration(self, maturity: float, spread: float, notional: float, frequency: int) -> float:
        """Calcule la duration de Macaulay (en années).
        Entre deux dates de reset, le coupon a déjà été fixé et ne bouge plus.
        Un choc de taux sur cette courte période fait donc varier la PV de ce coupon déjà déterminé (et du principal à maturité).
        """
        pv = self.price(maturity, spread, notional, frequency)
        r0 = self.vas.r0
        weighted = 0.0
        var_t = 1.0 / frequency
        for i in range(1, int(maturity * frequency) + 1):
            t = i * var_t
            fwd = self.forward_rate((i-1)*var_t, t)
            cf  = (fwd + spread) * var_t * notional if i <= maturity * frequency else notional
            weighted += t * cf * self.vas.price(r0, t)
        return weighted / pv

    def modified_duration(self, maturity: float, spread: float, notional: float, frequency: int) -> float:
        """Calcule la duration modifiée."""
        D_mac = self.macaulay_duration(maturity, spread, notional, frequency)
        return D_mac / (1 + self.vas.r0)

    def convexity(self, maturity: float, spread: float, notional: float, frequency: int) -> float:
        """Calcule la convexité (en années^2)."""
        pv = self.price(maturity, spread, notional, frequency)
        r0 = self.vas.r0
        weighted2 = 0.0
        var_t = 1.0 / frequency
        for i in range(1, int(maturity * frequency) + 1):
            t = i * var_t
            fwd = self.forward_rate((i-1)*var_t, t)
            cf  = (fwd + spread) * var_t * notional if i <= maturity * frequency else notional
            weighted2 += (t**2) * cf * self.vas.price(r0, t)
        return weighted2 / pv

    def dv01(self, maturity: float, spread: float, notional: float, frequency: int, bp: float = 1e-4) -> float:
        """Calcule le DV01 en bumpant r0 de +/-1bp."""
        orig = self.vas.r0
        # bump up
        self.vas.r0 = orig + bp
        pv_up = self.price(maturity, spread, notional, frequency)
        # bump down
        self.vas.r0 = orig - bp
        pv_down = self.price(maturity, spread, notional, frequency)
        # réinitialisation
        self.vas.r0 = orig
        return (pv_down - pv_up) / 2.0


if __name__ == '__main__':
    pricer = FloatingRateBondPricer(
        curve_file='data\RateCurve.xlsx',
        hist_file='data\data_vasicek.csv'
    )
    pricer.calibrate()

    maturity = 5
    spread = 0.0010
    notional = 100
    frequency = 2 

    print(f"Prix                = {pricer.price(maturity, spread, notional, frequency):.6f}")
    print(f"Duration Macaulay   = {pricer.macaulay_duration(maturity, spread, notional, frequency):.6f}")
    print(f"Duration modifiee   = {pricer.modified_duration(maturity, spread, notional, frequency):.6f}")
    print(f"Convexite           = {pricer.convexity(maturity, spread, notional, frequency):.6f}")
    print(f"DV01                = {pricer.dv01(maturity, spread, notional, frequency):.6f}")