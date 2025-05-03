import numpy as np
from BlackScholes_SVI import BSSVIModel

class OptionStrategiesPricer:
    """
    Pricer, risk metrics et stress tests pour
    stratégies optionnelles, basé sur BS + SVI.
    """
    def __init__(self, S, T, strikes, market_call_prices, r, option_type='call'):
        self.S0 = S
        self.T = T
        self.r = r
        self.model = BSSVIModel(S, T, strikes, market_call_prices, r, option_type=option_type)
        self.params = None

    def calibrate(self):
        """Calibre le modèle SVI sur les prix de marché."""
        self.params = self.model.calibrate()
        return self.params

    def _vol(self, K):
        a, b, rho, m, sigma = self.params
        k = np.log(K / self.S0)
        return self.model.svi_volatility(k, a, b, rho, m, sigma, self.T)

    def _price(self, K, option_type='call'):
        """Prix d'une vanille via SVI+BS."""
        sigma = self._vol(K)
        return self.model.bs_price(self.S0, K, self.T, self.r, sigma, option_type)

    def _greeks(self, K, option_type='call'):
        """Greeks (delta,gamma,Vega,theta,rho) d’une vanille via BS+SVI."""
        sigma = self._vol(K)
        return self.model.bs_greeks(self.S0, K, self.T, self.r, sigma, option_type)

    def _aggregate(self, legs):
        """
        Agrège price & greeks pour une liste de jambes.
        legs = [(qty, 'call'/'put'/'spot', K),...]
        Renvoie dict{price, delta, gamma, vega, theta, rho}.
        """
        res = dict(price=0., delta=0., gamma=0., vega=0., theta=0., rho=0.)
        for qty, typ, K in legs:
            if typ == 'spot':
                res['price'] += qty * (-self.S0)
                res['delta'] += qty * 1.0
            else:
                p = self._price(K, typ)
                delta, gamma, V, theta, rho = self._greeks(K, typ)
                res['price'] += qty * p
                res['delta'] += qty * delta
                res['gamma'] += qty * gamma
                res['vega'] += qty * V
                res['theta'] += qty * theta
                res['rho']   += qty * rho
        return res

    # --- Stratégies ---
    def call_spread(self, K1, K2):
        return self._aggregate([(1,'call',K1),(-1,'call',K2)])

    def put_spread(self, K1, K2):
        # Assure que K1 > K2 pour un bear put spread correct
        Kh, Kl = max(K1, K2), min(K1, K2)
        return self._aggregate([(1,'put',Kh),(-1,'put',Kl)])

    def butterfly(self, K1, K2, K3):
        return self._aggregate([(1,'call',K1),(-2,'call',K2),(1,'call',K3)])

    def collar(self, K_put, K_call):
        return self._aggregate([(1,'spot',None),(1,'put',K_put),(-1,'call',K_call)])

    def strip(self, K):
        return self._aggregate([(1,'call',K),(2,'put',K)])

    def strap(self, K):
        return self._aggregate([(2,'call',K),(1,'put',K)])

    # --- Payoff & risk metrics ---
    def payoff_profile(self, legs, S_min=0.5, S_max=1.5, points=1000):
        """P/L à maturité sur grille de spot."""
        Sgrid = np.linspace(S_min*self.S0, S_max*self.S0, points)
        cost = sum(q*self._price(K, typ) if typ!='spot' else q*(-self.S0)
                   for q,typ,K in legs)
        pl = np.zeros_like(Sgrid)
        for i,S in enumerate(Sgrid):
            pnl = 0.
            for qty,typ,K in legs:
                if typ=='spot': pnl += qty*(S - self.S0)
                elif typ=='call': pnl += qty*max(S-K,0)
                else: pnl += qty*max(K-S,0)
            pl[i] = pnl - cost
        return Sgrid, pl

    def _get_legs(self, strategy, *args):
        """Retourne les jambes pour une stratégie donnée."""
        if strategy == 'call_spread':
            return [(1,'call',args[0]),(-1,'call',args[1])]
        if strategy == 'put_spread':
            Kh, Kl = max(args[0], args[1]), min(args[0], args[1])
            return [(1,'put',Kh),(-1,'put',Kl)]
        if strategy == 'butterfly':
            return [(1,'call',args[0]),(-2,'call',args[1]),(1,'call',args[2])]
        if strategy == 'collar':
            return [(1,'spot',None),(1,'put',args[0]),(-1,'call',args[1])]
        if strategy == 'strip':
            return [(1,'call',args[0]),(2,'put',args[0])]
        if strategy == 'strap':
            return [(2,'call',args[0]),(1,'put',args[0])]
        raise ValueError(f"Stratégie inconnue: {strategy}")

    def break_even(self, strategy, *args):
        """Niveaux de spot où P/L=0."""
        legs = self._get_legs(strategy, *args)
        S, pl = self.payoff_profile(legs)
        idx = np.where(np.sign(pl[:-1]) != np.sign(pl[1:]))[0]
        be = []
        for i in idx:
            x0,x1 = S[i], S[i+1]; y0,y1 = pl[i], pl[i+1]
            be.append(x0 - y0*(x1-x0)/(y1-y0))
        return be

    def max_profit_loss(self, strategy, *args):
        """Max gain et perte à maturité."""
        legs = self._get_legs(strategy, *args)
        _, pl = self.payoff_profile(legs)
        return float(np.max(pl)), float(np.min(pl))

    def metrics(self, strategy, *args):
        """Retourne price, Greeks, BE, max P/L pour la stratégie."""
        agg = getattr(self, strategy)(*args)
        price = agg['price']
        greeks = {k: agg[k] for k in ['delta','gamma','vega','theta','rho']}
        be = self.break_even(strategy, *args)
        maxg, maxl = self.max_profit_loss(strategy, *args)
        stress = self.stress_test(strategy, *args)
        return { 'price': price,
                 'greeks': greeks,
                 'break_even': be,
                 'max_profit': maxg,
                 'max_loss': maxl,
                 'stress': stress }

    def stress_test(self, strategy, *args, shocks=[-0.1,-0.05,0.05,0.1]):
        legs = self._get_legs(strategy, *args)
        cost = sum(q*self._price(K,typ) if typ!='spot' else q*(-self.S0)
                   for q,typ,K in legs)
        results = {}
        for shock in shocks:
            S_stress = self.S0*(1+shock)
            pnl = sum(q*( (S_stress-self.S0) if typ=='spot'
                          else max(S_stress-K,0) if typ=='call'
                          else max(K-S_stress,0) )
                      for q,typ,K in legs)
            results[shock] = float(pnl - cost)
        return results

# === Exemple d'utilisation ===
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel("data\options_data.xlsx")
    strikes = df['Strike'].values
    calls = df['Call_Price'].values; puts = df['Put_Price'].values
    S, T = 100.0, 0.25
    r = np.mean([BSSVIModel.implied_rate(S,K,T,0.0,c,p)
                  for K,c,p in zip(strikes,calls,puts)])
    pricer = OptionStrategiesPricer(S,T,strikes,calls,r)
    pricer.calibrate()

    for name,args in [
        ('call_spread',(95,105)),
        ('put_spread',(105,95)),
        ('butterfly',(90,100,110)),
        ('collar',(95,105)),
        ('strip',(100,)),
        ('strap',(100,))
    ]:
        m = pricer.metrics(name,*args)
        print(f"\n== {name} ==")
        print(f"Prix           : {m['price']:.4f}")
        print(f"Greeks         : {m['greeks']}")
        print(f"Break-even     : {m['break_even']}")
        print(f"Max gain/loss  : {m['max_profit']:.4f} / {m['max_loss']:.4f}")
        print(f"Stress test    : {m['stress']}")
