# SwapPricer.py

from ObligationCoupon import CouponBondPricer
from VasicekModel import VasicekModel
from FloatingRateBond import FloatingRateBondPricer

class SwapPricer:
    def __init__(self,
                 curve_file: str,
                 hist_file: str,
                 notional: float,
                 fixed_rate: float,
                 maturity: float,
                 frequency: int = 2):
        self.N            = notional
        self.K            = fixed_rate
        self.T            = maturity
        self.freq         = frequency

        # vos pricers existants
        self.frn_pricer   = FloatingRateBondPricer(curve_file, hist_file)
        self.fix_pricer   = None

    def calibrate(self):
        # Calibrage NS + Vasicek pour la patte flottante
        self.frn_pricer.calibrate()

        # Récupère modèle Vasicek calibré et r0
        vas = self.frn_pricer.vas
        r0  = vas.r0

        # Initialise le pricer fixe sur la même base
        self.fix_pricer = CouponBondPricer(
            vasicek_model=vas,
            r0=r0,
            maturity=self.T,
            coupon_rate=self.K,
            face=self.N,
            freq=self.freq
        )

    def price(self, payer: str = 'fixed') -> float:
        PV_fix   = self.fix_pricer.price()
        PV_float = self.frn_pricer.price(
            maturity=self.T,
            spread=0.0,
            notional=self.N,
            frequency=self.freq
        )
        npv = PV_float - PV_fix
        return npv if payer.lower()=='fixed' else -npv

    def par_rate(self) -> float:
        """Taux fixe K* qui annule l'NPV."""
        vas      = self.frn_pricer.vas
        P        = lambda t: vas.price(vas.r0, t)
        delta        = 1.0 / self.freq
        n        = int(self.T * self.freq)
        annuity  = sum(delta * P(i*delta) for i in range(1, n+1))
        return (1 - P(self.T)) / annuity

    def pv01(self, bump: float = 1e-4) -> float:
        """
        Sensibilité PV01 : delta*NPV pour +1bp shift parallèle.
        Bump-and-reprice sur r0.
        """
        base = self.price()
        self._shift_r0(+bump)
        up   = self.price()
        self._shift_r0(-bump)
        return up - base

    def convexity(self, bump: float = 1e-4) -> float:
        """
        Convexité approximative :
        (V(+b)+V(-b)-2V(0)) / b^2
        """
        base = self.price()
        self._shift_r0(+bump)
        up   = self.price()
        self._shift_r0(-2*bump)
        down = self.price()
        self._shift_r0(+bump)  # restore
        return (up + down - 2*base) / (bump**2)

    def _shift_r0(self, delta: float):
        """Décale r0 pour les deux pricers."""
        self.frn_pricer.vas.r0 += delta
        self.fix_pricer.r0     += delta


if __name__ == '__main__':
    swap = SwapPricer(
        curve_file='data\RateCurve.xlsx',
        hist_file='data\data_vasicek.csv',
        notional=100.0,
        fixed_rate=0.05,
        maturity=3.0,
        frequency=2
    )
    swap.calibrate()
    print("NPV swap (payer fixe) :", swap.price())
    print("Par-rate             :", swap.par_rate())
    print("PV01  (1bp)          :", swap.pv01())
    print("Convexite            :", swap.convexity())
