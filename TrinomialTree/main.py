# S1
import numpy as np
import xlwings as xw
import datetime 
import time
from tree import Tree
from param import TimeParam
from MarketData import MarketData
from OptTrade import OptTrade2
from Greeks import Greeks
from OptBS import OptBS




if __name__ == '__main__':

    
    def OptionPricerBackwardPython(n_steps: int, start_date, mat_date,
                                start_price: float, rate: float, vol: float, div: float,
                                ex_date, opt_type: str, exer: str, k: float):
        """
        Fonction de calcul du prix de l'arbre trinomial
        """
        # Initialisation des classes utilisées pour le pricing
        market: MarketData = MarketData(start_price, rate, vol, div, ex_date)
        par: TimeParam = TimeParam(start_date, mat_date, n_steps)
        t: Tree = Tree(market, par)
        opt: OptTrade2 = OptTrade2(opt_type, exer, k)

        # Construction de l'arbre
        t.BuildTree()

        # récupération du prix de l'option
        return t.root.option_price_backard(opt)


    # Initialisation des instances utilisées pour le pricing


    # Arguments que l'utilisateur doit fournir : 
    start_price = 100
    rate = 0.15
    vol = 0.20
    div = 0
    ex_date = datetime.date(2024, 6, 15)
    start_date = datetime.date(2024, 1, 1)
    mat_date = datetime.date(2026, 1, 1)
    n_steps = 400
    opt_type= "Call"
    exer = "American"
    k = 110


    # # Calcul du prix de l'option via la méthode de pricing backward et du delta/gamma de l'option à la racine de l'arbre
    price: float = OptionPricerBackwardPython(n_steps, start_date, mat_date, start_price, rate, vol, div, ex_date, opt_type, exer, k)
    print(price)

    # Delta
    shift_price: float = 0.01*start_price
    price_up: float = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                            start_price + shift_price, rate, vol, div,
                                            ex_date, opt_type, exer, k)

    price_down: float = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                            start_price - shift_price, rate, vol, div,
                                            ex_date, opt_type, exer, k)

    delta = (price_up - price_down) / (2 * shift_price)
    print("Delta : ", delta)




    # Rho
    shift_rho: float = 0.01*rate
    price_up: float = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                            start_price, rate + shift_rho, vol, div,
                                            ex_date, opt_type, exer, k)

    price_down: float = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                            start_price, rate - shift_rho, vol, div,
                                            ex_date, opt_type, exer, k)

    rho = (price_up - price_down) / (2 * shift_rho * 100)
    print("Rho : ", rho)

    # Theta
    shift_day: int = 1
    shift = datetime.timedelta(days=shift_day)

    price_up: float = OptionPricerBackwardPython(n_steps, start_date, mat_date - shift,
                                start_price, rate, vol, div,
                                ex_date, opt_type, exer, k)

    price_down: float = OptionPricerBackwardPython(n_steps, start_date, mat_date + shift,
                                start_price, rate, vol, div,
                                ex_date, opt_type, exer, k)

    theta = (price_up - price_down)/(2*shift_day)
    print("Theta : ", theta)

    # Vega
    shift: float = vol * 0.01
    price_up: float = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                start_price, rate, vol + shift, div,
                                ex_date, opt_type, exer, k)

    price_down: float = OptionPricerBackwardPython(n_steps, start_date, mat_date,
                                start_price, rate, vol - shift, div,
                                ex_date, opt_type, exer, k)

    vega =  (price_up - price_down)/(2*shift*100)
    print("Vega : ", vega)