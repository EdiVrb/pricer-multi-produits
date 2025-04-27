import pandas as pd
import numpy as np
import datetime
import os, sys

from ObligationCoupon import VasicekModel, CouponBondPricer
from FloatingRateBond import FloatingRateBondPricer
from SwapPricer import SwapPricer
from BlackScholes_SVI import BSSVIModel
from OptionStrategies import OptionStrategiesPricer

# ─── importer TrinomialTree sans package ────────────────────────
BASE = os.path.dirname(__file__)
TT   = os.path.join(BASE, "TrinomialTree")
if TT not in sys.path:
    sys.path.insert(0, TT)

from TrinomialTree.MarketData import MarketData
from TrinomialTree.param import TimeParam
from TrinomialTree.tree import Tree
from TrinomialTree.OptTrade import OptTrade2

def price_american_option(n_steps, start_date, mat_date,
                           S0, rate, vol, div, ex_date,
                           opt_type, k):
    """Helper pour pricer une option américaine via un arbre trinomial."""
    market = MarketData(S0, rate, vol, div, ex_date)
    par = TimeParam(start_date, mat_date, n_steps)
    tree = Tree(market, par)
    opt = OptTrade2(opt_type, 'American', k)
    tree.BuildTree()
    return tree.root.option_price_backard(opt)


class PortfolioPricer:
    """
    Classe pour gérer un inventaire de positions multi-produits et calculer
    le pricing, les métriques de risque, et l'agrégation par bucket de maturité.

    Inventory CSV attendu avec colonnes:
      - ProductType: Zero-Coupon, Coupon, Floating, Swap, VanillaOption, OptionStrategy, AmericanOption
      - Strategy (pour OptionStrategy)
      - OptionType, Strike, Strike1, Strike2, Strike3, Maturity, Notional, Quantity,
        CouponRate, Spread, FixedRate, Frequency
    """
    def __init__(self,
                 inventory_file: str,
                 curve_file: str,
                 hist_file: str,
                 opt_mkt_file: str = None,
                 spot: float = None,
                 dividend: float = 0.0):
        self.inv_file = inventory_file
        self.curve_file = curve_file
        self.hist_file = hist_file
        self.opt_mkt_file = opt_mkt_file
        self.S0 = spot
        self.q = dividend

        self.df_inv = None
        self.vas = None
        self.frn = None
        self.svi_models = {}
        self.results = None

    def calibrate_fixed_income(self):
        self.frn = FloatingRateBondPricer(self.curve_file, self.hist_file)
        self.frn.calibrate()
        self.vas = self.frn.vas
        self.r0 = self.vas.r0

    def calibrate_svi(self):
        if not self.opt_mkt_file or self.S0 is None:
            return
        df_mkt = pd.read_excel(self.opt_mkt_file)
        strikes = df_mkt['Strike'].values
        calls = df_mkt['Call_Price'].values
        puts = df_mkt['Put_Price'].values
        maturities = sorted(df_mkt['Maturity'].unique()) if 'Maturity' in df_mkt else []
        if not maturities and self.df_inv is not None:
            maturities = sorted(self.df_inv['Maturity'].unique())
        for T in maturities:
            r_impl = np.mean([
                BSSVIModel.implied_rate(self.S0, K, T, self.q, c, p)
                for K, c, p in zip(strikes, calls, puts)
            ])
            model = BSSVIModel(self.S0, T, strikes, calls, r_impl)
            model.calibrate()
            self.svi_models[T] = (model, r_impl)

    def load_inventory(self):
        self.df_inv = pd.read_csv(self.inv_file)

    def price_positions(self, american_steps=200):
        rows = []
        for p in self.df_inv.itertuples(index=False):
            # Remplacer NaN dans Quantity par 1
            qty = p.Quantity if not pd.isna(p.Quantity) else 1
            # Remplacer NaN dans Quantity par 1
            qty = p.Quantity if not pd.isna(p.Quantity) else 1
            bucket = pd.cut([p.Maturity], bins=[0,1,2,5,30],
                            labels=["0-1","1-2","2-5",">5"])[0]
            price = dur = conv = None
            delta = gamma = vega = theta = rho = 0.0

            pt = p.ProductType.lower()
            if pt == 'zero-coupon':
                # Prix unitaire ZC * notional pour obtenir la valeur totale
                price = self.vas.price(self.r0, p.Maturity) * p.Notional
                dur = None
                conv = None
            elif pt == 'coupon':
                pr = CouponBondPricer(self.vas, self.r0,
                                       p.Maturity, p.CouponRate,
                                       face=p.Notional, freq=int(p.Frequency))
                price = pr.price()
                dur = pr.macaulay_duration()
                conv = pr.convexity()
            elif pt == 'floating':
                price = self.frn.price(p.Maturity, p.Spread,
                                       p.Notional, int(p.Frequency))
                dur = self.frn.macaulay_duration(p.Maturity, p.Spread,
                                                 p.Notional, int(p.Frequency))
                conv = self.frn.convexity(p.Maturity, p.Spread,
                                          p.Notional, int(p.Frequency))
            elif pt == 'swap':
                swap = SwapPricer(self.curve_file, self.hist_file,
                                   p.Notional, p.FixedRate,
                                   p.Maturity, int(p.Frequency))
                swap.calibrate()
                price = swap.price()
                conv = swap.convexity()
            elif pt == 'vanillaoption':
                model, r_impl = self.svi_models.get(p.Maturity, (None, None))
                if model:
                    sigma = model.svi_volatility(np.log(p.Strike/self.S0),
                                                 *model.params, p.Maturity)
                    price = BSSVIModel.bs_price(self.S0, p.Strike,
                                                p.Maturity, r_impl,
                                                sigma, p.OptionType)
                    d,g,v,th,ro = BSSVIModel.bs_greeks(self.S0, p.Strike,
                                                      p.Maturity, r_impl,
                                                      sigma,
                                                      option_type=p.OptionType)
                    price *= qty
                    delta, gamma, vega, theta, rho = [x * qty for x in (d,g,v,th,ro)]
            elif pt == 'optionstrategy':
                model, r_impl = self.svi_models.get(p.Maturity, (None, None))
                if model:
                    strikes = model.K
                    market_prices = model.market_prices
                    strat = OptionStrategiesPricer(self.S0, p.Maturity,
                                                   strikes, market_prices, r_impl)
                    strat.calibrate()
                    strat_name = p.Strategy
                    if strat_name in ('call_spread', 'put_spread', 'collar'):
                        args = (p.Strike1, p.Strike2)
                    elif strat_name == 'butterfly':
                        args = (p.Strike1, p.Strike2, p.Strike3)
                    elif strat_name in ('strip', 'strap'):
                        args = (p.Strike1,)
                    else:
                        args = ()
                    # récupère la métrique
                    m = strat.metrics(strat_name, *args)
                    price = m['price'] * qty
                    # greeks sont dans m['greeks']
                    delta = m['greeks']['delta'] * qty
                    gamma = m['greeks']['gamma'] * qty
                    vega  = m['greeks']['vega']  * qty
                    theta = m['greeks']['theta'] * qty
                    rho   = m['greeks']['rho']   * qty
            elif pt == 'americanoption':
                price = price_american_option(
                    american_steps,
                    datetime.datetime.today().date(),
                    datetime.datetime.today().date() + datetime.timedelta(days=int(p.Maturity*365)),
                    self.S0, self.q, p.Spread, self.q,
                    datetime.datetime.today().date() + datetime.timedelta(days=int(p.Maturity*365)),
                    p.OptionType.capitalize(), p.Strike)

            rows.append({
                'Product': p.ProductType,
                'Maturity': p.Maturity,
                'Notional': p.Notional,
                'Quantity': qty,
                'Price': price,
                'Duration': dur,
                'Convexity': conv,
                'Delta': delta,
                'Gamma': gamma,
                'Vega': vega,
                'Theta': theta,
                'Rho': rho,
                'Bucket': bucket
            })
        self.results = pd.DataFrame(rows)
        return self.results

    def aggregate(self):
        if self.results is None:
            return None
        # Agrégation par bucket : somme des valeurs absolues et averages pondérées pour Duration/Convexity
        def weighted_avg(series):
            mask = series.notna()
            if not mask.any():
                return None
            weights = self.results.loc[series.index, 'Price'][mask]
            return np.average(series[mask], weights=weights)

        agg = self.results.groupby('Bucket').agg({
            'Price':    'sum',
            'Notional': 'sum',
            'Quantity': 'sum',
            'Delta':    'sum',
            'Gamma':    'sum',
            'Vega':     'sum',
            'Theta':    'sum',
            'Rho':      'sum',
            'Duration': weighted_avg,
            'Convexity':weighted_avg
        }).reset_index()
        return agg


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation du pricer
    pf = PortfolioPricer(
        inventory_file="data\portfolio_inventory.csv",
        curve_file="data\RateCurve.xlsx",
        hist_file="data\data_vasicek.csv",
        opt_mkt_file="data\options_data.xlsx",
        spot=100.0,
        dividend=0.0
    )
    # Vérification de l'existence du fichier d'inventaire
    if not os.path.isfile(pf.inv_file):
        print(f"Fichier d'inventaire introuvable: {pf.inv_file}"
              "Merci de placer votre CSV 'portfolio_inventory.csv' dans le répertoire courant ou de spécifier le chemin correct.")
        sys.exit(1)

    # Chargement des données et calibrages
    pf.load_inventory()
    pf.calibrate_fixed_income()
    pf.calibrate_svi()

    # Calcul des prix et agrégation
    df_res = pf.price_positions(american_steps=300)
    df_agg = pf.aggregate()

    # Affichage des résultats
    print("Détail des positions :")
    print(df_res.to_string(index=False))
    print("Agrégation par bucket de maturité :")
    print(df_agg.to_string(index=False))
