# app.py

import os
import sys
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Réduction globale de la taille de police pour les graphiques
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# ─── Hack pour importer TrinomialTree sans package ────────────────────────
BASE = os.path.dirname(__file__)
TT   = os.path.join(BASE, "TrinomialTree")
if TT not in sys.path:
    sys.path.insert(0, TT)

# ─── Vos modules existants et nouveaux produits ──────────────────────────
from ObligationCoupon        import VasicekModel, CouponBondPricer
from BlackScholes_SVI        import BSSVIModel
from FloatingRateBond        import FloatingRateBondPricer
from SwapPricer              import SwapPricer
from OptionStrategies        import OptionStrategiesPricer
from Autocall                import AutocallParameters, AutocallType, AutocallPricer, plot_autocall_simulation
from Portfolio               import PortfolioPricer  # <-- Intégration Portfolio

# ─── Modules de l'arbre trinomial pour l'option américaine ─────────────
from TrinomialTree.MarketData import MarketData
from TrinomialTree.param      import TimeParam
from TrinomialTree.tree       import Tree
from TrinomialTree.OptTrade   import OptTrade2


def OptionPricerBackwardPython(n_steps: int,
                               start_date: datetime.date,
                               mat_date:   datetime.date,
                               start_price: float,
                               rate:        float,
                               vol:         float,
                               div:         float,
                               ex_date:     datetime.date,
                               opt_type:    str,
                               exer:        str,
                               k:           float) -> float:
    """Construit l'arbre trinomial et renvoie le prix backward de l'option."""
    market = MarketData(start_price, rate, vol, div, ex_date)
    par    = TimeParam(start_date, mat_date, n_steps)
    tree   = Tree(market, par)
    opt    = OptTrade2(opt_type, exer, k)
    tree.BuildTree()
    return tree.root.option_price_backard(opt)


# ─── Configuration Streamlit ─────────────────────────────────────────────
st.set_page_config(page_title="Pricer Multi-Produits", layout="wide")
st.title("💰 Pricer Multi-Produits")

menu = [
    "Zero-Coupon Bond",
    "Coupon Bond",
    "Floating Rate Bond",
    "Swap",
    "European Option",
    "Option Strategies",
    "American Option",
    "Autocall",
    "Portfolio"    # <-- Nouvel onglet Portfolio
]
choice = st.sidebar.selectbox("Produit", menu)


# ─── Zero-Coupon Bond ────────────────────────────────────────────────────
if choice == "Zero-Coupon Bond":
    st.header("📈 Zero-Coupon (Vasicek)")
    uploaded = st.file_uploader("CSV historique (Date, Dernier)", type="csv")
    dt       = st.number_input("dt (ans)", 1/52., format="%.4f")
    margin   = st.number_input("Spread", 0.0, format="%.4f")
    if uploaded:
        df = pd.read_csv(uploaded, sep=";", parse_dates=["Date"], dayfirst=True)
        df = df.sort_values("Date").reset_index(drop=True)
        df["Rate"] = df["Dernier"].str.replace(",", ".").astype(float) / 100
        r0 = df["Rate"].iloc[-1]
        T  = st.number_input("Maturité ZC (ans)", 5.0, format="%.2f")
        @st.cache(suppress_st_warning=True, show_spinner=False)
        def calib_vas(rates, dt, margin):
            m = VasicekModel()
            m.calibrate(rates, dt, margin)
            return m
        vas = calib_vas(df["Rate"].values, dt, margin)
        st.subheader("Paramètres Vasicek calibrés")
        st.write({"a": vas.a, "k": vas.k, "σ": vas.sigma})
        st.metric("Prix Zéro-Coupon", f"{vas.price(r0, T):.6f}")


# ─── Coupon Bond ─────────────────────────────────────────────────────────
elif choice == "Coupon Bond":
    st.header("🏦 Obligation à coupons")
    uploaded = st.file_uploader("CSV historique (Date, Dernier)", type="csv", key="cb")
    dt       = st.number_input("dt (ans)", 1/52., format="%.4f", key="cb_dt")
    margin   = st.number_input("Spread", 0.0, format="%.4f", key="cb_mg")
    if uploaded:
        df = pd.read_csv(uploaded, sep=";", parse_dates=["Date"], dayfirst=True)
        df = df.sort_values("Date").reset_index(drop=True)
        df["Rate"] = df["Dernier"].str.replace(",", ".").astype(float) / 100
        r0 = df["Rate"].iloc[-1]
        @st.cache(suppress_st_warning=True, show_spinner=False)
        def calib_vas(rates, dt, margin):
            m = VasicekModel()
            m.calibrate(rates, dt, margin)
            return m
        vas = calib_vas(df["Rate"].values, dt, margin)
        st.subheader("Paramètres Vasicek calibrés")
        st.write({"a": vas.a, "k": vas.k, "σ": vas.sigma})
        T       = st.number_input("Maturité (ans)", 5.0, format="%.2f")
        coup_ann= st.number_input("Coupon annuel (%)", 2.5, format="%.2f") / 100
        nominal = st.number_input("Notional", 100.0)
        freq    = st.selectbox("Fréquence coupons", [1, 2, 4, 12])
        pr      = CouponBondPricer(vas, r0, T, coup_ann, nominal, freq)
        st.metric("Prix obligation", f"{pr.price():.6f}")
        st.write(f"- Duration : **{pr.macaulay_duration():.4f} ans**  \n"
                 f"- Convexité : **{pr.convexity():.4f} ans²**")


# ─── Floating Rate Bond ──────────────────────────────────────────────────
elif choice == "Floating Rate Bond":
    st.header("💧 Floating Rate Bond")
    curve = st.file_uploader("Excel Nelson-Siegel (Rate Curve)", type="xlsx", key="frn_curve")
    hist  = st.file_uploader("CSV historique taux court", type="csv", key="frn_hist")
    if curve and hist:
        frn = FloatingRateBondPricer(curve, hist)
        frn.calibrate()
        T        = st.number_input("Maturité (ans)", 5.0, format="%.2f")
        spread   = st.number_input("Spread (%)", 0.1, format="%.2f") / 100
        notional = st.number_input("Notional", 100.0)
        freq     = st.selectbox("Fréquence reset", [1, 2, 4, 12], key="frn_freq")
        price    = frn.price(T, spread, notional, freq)
        dur_mac  = frn.macaulay_duration(T, spread, notional, freq)
        dur_mod  = frn.modified_duration(T, spread, notional, freq)
        conv     = frn.convexity(T, spread, notional, freq)
        dv01     = frn.dv01(T, spread, notional, freq)
        st.metric("Prix FRN", f"{price:.6f}")
        st.write({
            "Duration Macaulay":   dur_mac,
            "Duration modifiée":   dur_mod,
            "Convexité":           conv,
            "DV01":                dv01
        })


# ─── Swap ────────────────────────────────────────────────────────────────
elif choice == "Swap":
    st.header("🔁 Interest Rate Swap")
    curve = st.file_uploader("Excel Nelson-Siegel (Rate Curve)", type="xlsx", key="swap_curve")
    hist  = st.file_uploader("CSV historique taux court", type="csv", key="swap_hist")
    if curve and hist:
        notional   = st.number_input("Notional", 100.0)
        fixed_rate = st.number_input("Fixed Rate (%)", 5.0, format="%.2f") / 100
        T          = st.number_input("Maturité (ans)", 5.0, format="%.2f")
        freq       = st.selectbox("Fréquence coupons", [1, 2, 4, 12], key="swap_freq")
        swap = SwapPricer(curve, hist, notional, fixed_rate, T, freq)
        swap.calibrate()
        npv   = swap.price()
        par   = swap.par_rate()
        pv01  = swap.pv01()
        conv  = swap.convexity()
        st.metric("NPV (payer fixe)", f"{npv:.6f}")
        st.write({
            "Par-rate":  par,
            "PV01":      pv01,
            "Convexité": conv
        })


# ─── European Option ─────────────────────────────────────────────────────
elif choice == "European Option":
    st.header("🗡️ Option Européenne")
    mode = st.sidebar.radio("Mode", ["BS sans calibration", "BS avec calibration (SVI)"], key="euro_mode")
    if mode == "BS sans calibration":
        S    = st.number_input("Spot S", 100.0)
        K    = st.number_input("Strike K", 100.0)
        T    = st.number_input("Maturité (ans)", 1.0, format="%.2f")
        r    = st.number_input("r (%)", 0.0, format="%.2f") / 100
        σ    = st.slider("σ (%)", 0.0, 200.0, 20.0) / 100
        typ  = st.selectbox("Type", ["call", "put"])
        if st.button("→ Prix & Greeks BS"):
            p, d, g, v, th, ro = (
                BSSVIModel.bs_price(S, K, T, r, σ, option_type=typ),
                *BSSVIModel.bs_greeks(S, K, T, r, σ, option_type=typ)
            )
            st.metric("Prix BS", f"{p:.4f}")
            st.write({"Δ": d, "Γ": g, "ν": v, "θ": th, "ρ": ro})
    else:
        xlsx = st.file_uploader("Excel Strike/Call_Price/Put_Price", type="xlsx", key="euro_svi")
        S    = st.number_input("Spot S", 100.0, key="euro_s")
        T    = st.number_input("T (ans)", 0.25, format="%.2f", key="euro_T")
        q    = st.number_input("q (%)", 0.0, format="%.2f", key="euro_q") / 100
        if xlsx:
            df = pd.read_excel(xlsx)
            strikes = df["Strike"].values
            calls   = df["Call_Price"].values
            puts    = df["Put_Price"].values
            # taux implicite
            r_impl = np.mean([
                BSSVIModel.implied_rate(S, K0, T, q, c, p)
                for K0, c, p in zip(strikes, calls, puts)
            ])
            st.write("r implicite :", f"{r_impl:.4f}")
            # calibration SVI
            model  = BSSVIModel(S, T, strikes, calls, r_impl)
            params = model.calibrate()
            st.write("Params SVI :", np.round(params, 4))
            # grid smile
            K_min, K_max = strikes.min(), strikes.max()
            K_grid = np.linspace(K_min, K_max, 50)
            logm   = np.log(K_grid / S)
            vol_grid = model.svi_volatility(logm, *params, T)
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            ax.plot(K_grid, vol_grid, lw=2, label="Smile SVI")
            ax.scatter(strikes, model.get_vols_and_prices()[0], color="red", label="Vol marché")
            ax.set_xlabel("Strike"); ax.set_ylabel("Vol implicite")
            ax.set_title("Smile de volatilité implicite"); ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            # pricing + greeks
            K0  = st.number_input("Strike pour pricing", float(strikes[0]), key="euro_K0")
            typ = st.selectbox("Type", ["call", "put"], key="euro_typ")
            if st.button("→ Price & Greeks SVI+BS"):
                vol0 = model.svi_volatility(np.log(K0/S), *params, T)
                p0, d, g, v, th, ro = (
                    BSSVIModel.bs_price(S, K0, T, r_impl, vol0, option_type=typ),
                    *BSSVIModel.bs_greeks(S, K0, T, r_impl, vol0, option_type=typ)
                )
                st.metric(f"Prix K={K0}", f"{p0:.4f}")
                st.write({"Δ": d, "Γ": g, "ν": v, "θ": th, "ρ": ro})


# ─── Option Strategies ───────────────────────────────────────────────────
elif choice == "Option Strategies":
    st.header("🧮 Strategies Optionnelles (BS + SVI)")
    xlsx = st.file_uploader("Excel Strike/Call_Price/Put_Price", type="xlsx", key="strat_file")
    if xlsx:
        df = pd.read_excel(xlsx)
        sorted_strikes = np.sort(df["Strike"].values)
        S = st.number_input("Spot S", 100.0, key="strat_s")
        T = st.number_input("T (ans)", 0.25, format="%.2f", key="strat_T")
        r = st.number_input("r (%)", 0.0, format="%.2f", key="strat_r") / 100
        pricer = OptionStrategiesPricer(S, T, sorted_strikes, df["Call_Price"].values, r)
        pricer.calibrate()
        strat = st.sidebar.selectbox("Stratégie", ["call_spread","put_spread","butterfly","collar","strip","strap"], key="strat_choice")
        # champs dynamiques
        if strat in ("call_spread","put_spread"):
            K1 = st.number_input("Strike 1", value=sorted_strikes[0], key="K1")
            K2 = st.number_input("Strike 2", value=sorted_strikes[-1], key="K2")
            args=(K1,K2)
        elif strat=="butterfly":
            K1=st.number_input("K1",value=sorted_strikes[0],key="b1")
            K2=st.number_input("K2",value=sorted_strikes[len(sorted_strikes)//2],key="b2")
            K3=st.number_input("K3",value=sorted_strikes[-1],key="b3")
            args=(K1,K2,K3)
        elif strat=="collar":
            Kp=st.number_input("Strike Put",value=sorted_strikes[0],key="cp")
            Kc=st.number_input("Strike Call",value=sorted_strikes[-1],key="cc")
            args=(Kp,Kc)
        else:
            K0=st.number_input("Strike",value=sorted_strikes[0],key="s0")
            args=(K0,)
        if st.button("→ Calculer metrics", key="strat_go"):
            m=pricer.metrics(strat,*args)
            st.metric("Prix",f"{m['price']:.4f}")
            st.write("Greeks",m["greeks"])
            st.write("Break-even",m["break_even"])
            st.write("Max gain",m["max_profit"])
            st.write("Max perte",m["max_loss"])
            st.write("Stress test",m["stress"])
            legs=Sgrid=None; pl=None
            legs = pricer._get_legs(strat,*args)
            Sgrid,pl = pricer.payoff_profile(legs)
            fig,ax=plt.subplots(figsize=(6,3),dpi=100)
            ax.plot(Sgrid,pl,lw=2)
            ax.set_title("Payoff à maturité")
            ax.set_xlabel("Spot"); ax.set_ylabel("P/L")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)


# ─── American Option ────────────────────────────────────────────────────
elif choice == "American Option":
    st.header("🦅 Option Américaine (Trinomial)")
    n_steps    = st.number_input("Pas (n_steps)", 400, step=1)
    start_date = st.date_input("Date départ", datetime.date.today())
    mat_date   = st.date_input("Date maturité", start_date + datetime.timedelta(days=365))
    ex_date    = st.date_input("Date exercice anticipé", start_date + datetime.timedelta(days=180))
    S0   = st.number_input("Spot S0", 100.0)
    rate = st.number_input("r (%)", 0.0, format="%.2f") / 100
    vol  = st.number_input("σ (%)", 20.0, format="%.2f") / 100
    div  = st.number_input("q (%)", 0.0, format="%.2f") / 100
    opt_type = st.selectbox("Type", ["Call", "Put"])
    k        = st.number_input("Strike K", 100.0)
    if st.button("→ Prix & Greeks Américaines"):
        price = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                           S0, rate, vol, div,
                                           ex_date, opt_type, "American", k)
        # Greeks via différences finies
        shift_S = 0.01 * S0
        pu = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                        S0+shift_S, rate, vol, div,
                                        ex_date, opt_type, "American", k)
        pd = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                        S0-shift_S, rate, vol, div,
                                        ex_date, opt_type, "American", k)
        delta = (pu - pd) / (2 * shift_S)
        # Rho
        bump_r = 1e-4
        ru = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                        S0, rate+bump_r, vol, div,
                                        ex_date, opt_type, "American", k)
        rd = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                        S0, rate-bump_r, vol, div,
                                        ex_date, opt_type, "American", k)
        rho = (ru - rd) / (2 * bump_r)
        # Theta
        shift_t = datetime.timedelta(days=1)
        tu = OptionPricerBackwardPython(n_steps, start_date, mat_date-shift_t,
                                        S0, rate, vol, div,
                                        ex_date, opt_type, "American", k)
        td = OptionPricerBackwardPython(n_steps, start_date, mat_date+shift_t,
                                        S0, rate, vol, div,
                                        ex_date, opt_type, "American", k)
        theta = (tu - td) / 2
        # Vega
        bump_v = 0.01 * vol
        vu = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                        S0, rate, vol+bump_v, div,
                                        ex_date, opt_type, "American", k)
        vd = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                        S0, rate, vol-bump_v, div,
                                        ex_date, opt_type, "American", k)
        vega = (vu - vd) / (2 * bump_v)
        st.metric("Prix Américain", f"{price:.4f}")
        st.write({"Δ": delta, "ρ": rho, "θ": theta, "ν": vega})


# ─── Auto Call ────────────────────────────────────────────────────
elif choice == "Autocall":
    st.header("📈 Autocall")
    # Sélection du type
    ac_type = st.selectbox("Type d'Autocall", [t.value for t in AutocallType])
    # Paramètres de base
    S0      = st.number_input("Spot initial", 100.0)
    strike_pct = st.number_input("% Strike (en fraction)", 1.0)
    barrier_pct= st.number_input("% Barrière (en fraction)", 0.6)
    coupon_rate = st.number_input("Taux coupon annuel (%)", 7.0) / 100
    vol     = st.number_input("Volatilité (%)", 20.0) / 100
    r       = st.number_input("Taux sans risque (%)", 3.0) / 100
    q       = st.number_input("Dividende (%)", 1.0) / 100
    T       = st.number_input("Maturité (ans)", 5.0)
    obs_dates_str = st.text_input("Dates d'observation (ans, séparées par des virgules)", "1,2,3,4,5")
    autocall_barriers_str = st.text_input("Barrières Autocall (fraction, séparées par virgules)", "1,0.95,0.9,0.85,0.8")
    memory = False
    if ac_type == AutocallType.PHOENIX.value:
        memory = st.checkbox("Effet mémoire (Phoenix)")
        coupon_barriers_str = st.text_input("Barrières coupon (fraction, séparées par des virgules)", "0.8,0.8,0.8,0.8,0.8")

    if st.button("→ Calculer Autocall"):
        # Parsing des listes
        obs_dates = [float(x) for x in obs_dates_str.split(",")]
        autob = [float(x) for x in autocall_barriers_str.split(",")]
        coup_bar = None
        if ac_type == AutocallType.PHOENIX.value:
            coup_bar = [float(x) for x in coupon_barriers_str.split(",")]

        params = AutocallParameters(
            underlying_spot=S0,
            strike_percentage=strike_pct,
            barrier_percentage=barrier_pct,
            coupon_rate=coupon_rate,
            volatility=vol,
            risk_free_rate=r,
            dividend_yield=q,
            maturity=T,
            observation_dates=obs_dates,
            autocall_barriers=autob,
            coupon_barriers=coup_bar,
            memory_effect=memory
        )
        pricer = AutocallPricer(AutocallType(ac_type))
        result = pricer.price(params, nb_simulations=20000)
        st.subheader("Résultats")
        st.json(result)
        # Graphique d'exemple
        fig = plot_autocall_simulation(pricer, params, nb_paths=5)
        st.pyplot(fig)

# ─── Portfolio ────────────────────────────────────────────────────────────
elif choice == "Portfolio":
    st.header("📊 Portfolio Multi-Produits")
    inv_file   = st.file_uploader("1) Inventaire CSV", type="csv", key="pf_inv")
    curve_file = st.file_uploader("2) Curve NS (Excel)", type="xlsx", key="pf_curve")
    hist_file  = st.file_uploader("3) Historique taux court (CSV)", type="csv", key="pf_hist")
    opt_file   = st.file_uploader("4) Options marché (Excel)", type="xlsx", key="pf_opt")
    S0         = st.number_input("Spot initial S₀", 100.0, key="pf_spot")
    div        = st.number_input("Dividendes q (%)", 0.0, format="%.2f", key="pf_div") / 100

    if st.button("→ Lancer pricing Portfolio", key="pf_go"):
        if not (inv_file and curve_file and hist_file):
            st.error("Veuillez uploader les fichiers inventaire, curve et historique.")
        else:
            # Sauvegarde temporaire
            inv_path   = inv_file.name
            curve_path = curve_file.name
            hist_path  = hist_file.name
            opt_path   = opt_file.name if opt_file else None

            # Instanciation
            pf = PortfolioPricer(
                inventory_file=inv_path,
                curve_file=curve_path,
                hist_file=hist_path,
                opt_mkt_file=opt_path,
                spot=S0,
                dividend=div
            )
            pf.load_inventory()
            pf.calibrate_fixed_income()
            pf.calibrate_svi()
            df_res = pf.price_positions(american_steps=200)
            df_agg = pf.aggregate()

            st.subheader("Détail des positions")
            st.dataframe(df_res)

            st.subheader("Agrégation par bucket de maturité")
            st.dataframe(df_agg)
