import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import pandas as pd
from datetime import datetime, date
from numpy.polynomial.laguerre import lagfit, lagval
from numpy.polynomial.hermite import hermfit, hermval


class Brownian:
    def __init__(self, t: float, steps: int, Nb_Simulations: int, seed: int | None = None) -> None:
        self.t = t
        self.steps = steps
        self.Nb_Simulations = Nb_Simulations

        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> np.array:
        times = np.linspace(0, self.t, self.steps)
        dt = times[1] - times[0]
        #dt = self.t/self.steps
        increments = np.random.randn(self.Nb_Simulations, self.steps - 1) * np.sqrt(dt)
        #increments = np.random.normal(0, np.sqrt(dt), (self.Nb_Simulations, self.steps - 1))
        B0 = np.zeros((self.Nb_Simulations, 1))
        brownian_paths = np.concatenate((B0, np.cumsum(increments, axis=1)), axis=1)
        return brownian_paths, times
    

class Market:
    def __init__(self, S, r, q, sigma):
        self.S = S
        self.r = r
        self.q = q
        self.sigma = sigma

    def discount_value(self, value, T):
            return value * np.exp(-self.r * T)
    

class Option:
    def __init__(self, K, option_type: str, maturity_date : date, pricing_date : date, option_maturity : str):
        self.K = K
        self.option_type = option_type.lower()
        self.option_maturity = option_maturity.lower()
        self.maturity_date = maturity_date
        self.pricing_date = pricing_date

        self.maturity : float = (self.maturity_date - self.pricing_date).days / 365

    def compute_payoff(self, ST):

        payoff = np.zeros_like(ST)

        if self.option_maturity == "europeenne":

            if self.option_type == "call":

                payoff[:, -1] =  np.maximum(ST[:, -1]   - self.K, 0.0)
            else:  
                payoff[:, -1]= np.maximum(self.K - ST[:, -1]  , 0.0)
        else:
            if self.option_type == "call":
                payoff =  np.maximum(ST - self.K, 0.0)
            else:
                payoff =  np.maximum(self.K - ST, 0.0)

        return payoff 

class MonteCarloSimulation:

    def __init__(self, market: Market, option: Option, brownian: Brownian, Nb_Simulations: int):
        self.Nb_Simulations = Nb_Simulations
        self.market = market
        self.option = option
        self.brownian = brownian
     

    def simulate_paths(self):

        W, times = self.brownian.generate() 
        
        S_paths = np.zeros_like(W)  
        
        exponent = (self.market.r - self.market.q - 0.5 * self.market.sigma**2)*times\
                           + self.market.sigma * W
        
        S_paths = self.market.S * np.exp(exponent)

        #steps = len(times)
        #for i in range(self.Nb_Simulations):
        #    for j in range(steps):
        #        t_j = times[j]
                
        #        exponent = (self.market.r - self.market.q - 0.5 * self.market.sigma**2)*t_j \
        #                   + self.market.sigma * W[i, j]
        #        S_paths[i, j] = self.market.S * np.exp(exponent)

        return S_paths, times


    def simulate(self) :
                
        S_paths, _ = self.simulate_paths()
        payoff = self.option.compute_payoff(S_paths)
        
        #St = self.market.S * np.exp((self.market.r - self.market.q - 0.5 * self.market.sigma**2) * self.T + self.market.sigma * W)
        #payoff = self.option.compute_payoff(St)

        payoff_final = payoff[:, -1]  
        price = np.exp(-self.market.r * self.option.maturity) * np.mean(payoff_final)
        return price

    def Regression(self, X: np.ndarray, Y: np.ndarray, poly_degree: int, poly_type) -> np.ndarray:
        if poly_type == "standard":
            return np.polyval(np.polyfit(X, Y, poly_degree), X)
        elif poly_type == "laguerre":
            return lagval(X, lagfit(X, Y, poly_degree))
        elif poly_type == "hermite":
            return hermval(X, hermfit(X, Y, poly_degree))
        else:
            raise ValueError(f"Erreur dans type de poly: {poly_type}")

            
    def LSM(self, poly_degree, poly_type):
        
        S_paths, times = self.simulate_paths()
        steps = len(times)
        dt = times[1] - times[0]  
        r = self.market.r
        
        payoff = self.option.compute_payoff(S_paths)

        CF = np.zeros_like(payoff)
        CF[:,-1] = payoff[:,-1]

        for j in range(steps - 2, -1, -1):
            discounted_CF_next = CF[:,j+1] * np.exp(-r * dt)
            in_the_money = payoff[:,j] > 0

            CF[:,j] = discounted_CF_next 
            
            if np.any(in_the_money):
                X = S_paths[in_the_money, j]             
                Y = discounted_CF_next[in_the_money]
                                                                                                    #faire fonction regression de cette partie
                #X_design = np.column_stack([X**p for p in range(poly_degree+1)])
                
                #linreg = np.polyfit(X, Y, poly_degree)
                #continuation_values = np.polyval(linreg,X)

                continuation_values = self.Regression(X,Y,poly_degree, poly_type)

                exercise_now = payoff[in_the_money, j] > continuation_values

                CF[in_the_money, j] = np.where(exercise_now,
                                               payoff[in_the_money, j],  
                                               discounted_CF_next[in_the_money])
        
                #exercised_indices = np.where(in_the_money)[0][exercise_now]
                #for idx in exercised_indices:
                #    CF[idx, j+1:] = 0.0
            else:
                
                CF[:,j] = discounted_CF_next                 

        price = np.mean(CF[:,0])
        return price

class BlackScholes:
    def __init__(self, market: Market, option: Option):
        self.market = market
        self.option = option
        #self.T = T

    def pricer(self) -> float:
        d1 = (math.log(self.market.S / self.option.K) + (self.market.r - self.market.q + 0.5 * self.market.sigma**2) * self.option.maturity) / (
            self.market.sigma * math.sqrt(self.option.maturity)
        )
        d2 = d1 - self.market.sigma * math.sqrt(self.option.maturity)

        if self.option.option_type == "call":
            price = self.market.S * math.exp(-self.market.q * self.option.maturity) * norm.cdf(d1) - \
                    self.option.K * math.exp(-self.market.r * self.option.maturity) * norm.cdf(d2)
        else:  
            price = self.option.K * math.exp(-self.market.r * self.option.maturity) * norm.cdf(-d2) - \
                    self.market.S * math.exp(-self.market.q * self.option.maturity) * norm.cdf(-d1)
        return price




def convergence_price(market: Market, option: Option, seed: int, steps: int, max_simulations: int) -> pd.DataFrame:
    bs = BlackScholes(market, option)
    price_bs = bs.pricer()

    convergence_data = []

    for nb_simulations in range(100, max_simulations + 1, 200):
        brownian = Brownian(option.maturity, steps, nb_simulations, seed)
        mc_sim = MonteCarloSimulation(market, option, brownian, nb_simulations)
        mc_price = mc_sim.simulate()

        mc_lsm = mc_sim.LSM(poly_degree=5, poly_type= "standard")

        gap_euro = (mc_price - price_bs) * np.sqrt(nb_simulations)
        gap_lsm = (mc_lsm - price_bs) * np.sqrt(nb_simulations)

        convergence_data.append([nb_simulations, mc_price, mc_lsm,  gap_euro, gap_lsm])

    df = pd.DataFrame(convergence_data, columns=["Nb_Simulations", "MC_Price", "MC_LSM", "GapE", "GapA"])

    plt.figure(figsize=(10, 6))
    plt.plot(df["Nb_Simulations"], df["MC_Price"], label="Monte Carlo Price", marker='o')
    plt.plot(df["Nb_Simulations"], df["MC_LSM"], label="Least Squares Monte Carlo (LSM)", marker='x')
    plt.axhline(y=price_bs, color="r", linestyle="--", label="Black-Scholes Price")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price")
    plt.title("Monte Carlo Convergence vs Black-Scholes")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df



"""def convergence_ecartype(market: Market, option: Option, seed: int, steps: int, max_simulations: int) -> pd.DataFrame:
    convergence_data = []

    for nb_simulations in range(1, max_simulations + 1, 10):
        prices = []

        for _ in range(nb_simulations):
            brownian = Brownian(option.maturity, steps, nb_simulations, seed)
            mc_sim = MonteCarloSimulation(market, option, brownian, nb_simulations)
            price_LSM = mc_sim.LSM(poly_degree=2)
            prices.append(price_LSM)

        price_average = np.mean(prices)
        price_std = np.std(prices)

        convergence_data.append([nb_simulations, price_average, price_std])

    df = pd.DataFrame(convergence_data, columns=["Nb_Simulations", "Prix_Moyen", "Ecart_Type"])

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x=df["Nb_Simulations"], 
        y=df["Prix_Moyen"], 
        yerr=2.0 * df["Ecart_Type"], 
        fmt='o-', 
        ecolor='red', 
        capsize=3
    )
    plt.xscale("log")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Option Price")
    plt.title("Convergence of MC Price (+/- std dev)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.show()

    return df"""

        

# S = 100
S = 105
seed = 2
steps = 10
Nb_Simulations = 1000000
r = 0.06
q = 0.0
sigma = 0.2
K = 100
poly_type = "standard"

degree = 2
OptionMaturity = "americaine"
pricing_date = datetime.strptime("01/01/2024", "%d/%m/%Y").date()
maturity_date = datetime.strptime("01/01/2025", "%d/%m/%Y").date()

option = Option(K, "put",maturity_date , pricing_date, OptionMaturity)
maturity = option.maturity
brownian = Brownian(maturity, steps, Nb_Simulations, seed)
market = Market(S, r, q, sigma)

mc_sim = MonteCarloSimulation(market, option, brownian, Nb_Simulations)

max_simulations = 300

price = mc_sim.simulate()
print(f"Option Price: {price:.4f}")

price_am = mc_sim.LSM(poly_degree=2, poly_type="standard")
print(f"Prix de l'option (LSM) : {price_am: .4f}")

bs_model = BlackScholes(market, option)
bs_price = bs_model.pricer()
print(f"Prix de l'option (Black-Scholes): {bs_price:.4f}")

#df_convergence = convergence_price(market, option, seed, steps, max_simulations)
#print(df_convergence)

#df_convergence_average = convergence_ecartype(market, option, seed, steps, max_simulations)
#print(df_convergence_average)



class Greeks:
    def __init__(self, market : Market, option : Option, mc_sim : MonteCarloSimulation, epsilon : float):
        self.market = market
        self.option = option
        self.mc_sim = mc_sim
        self.epsilon = epsilon


    def calculate_delta(self) -> float:
        
        original_S = self.market.S

        self.market.S = original_S * (1 + self.epsilon)
        price_up = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

        self.market.S = original_S* (1 - self.epsilon)
        price_down = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

        self.market.S = original_S

        delta = (price_up - price_down) / (2 * original_S * self.epsilon)

        return delta

    def calculate_gamma(self) -> float:
   
        original_S = self.market.S

    
        dS = original_S * self.epsilon

  
        price_0 = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

 
        self.market.S = original_S + dS
        price_up = self.mc_sim.LSM(poly_degree=2, poly_type="standard")


        self.market.S = original_S - dS
        price_down = self.mc_sim.LSM(poly_degree=2, poly_type="standard")


        self.market.S = original_S

   
        gamma = (price_up - 2 * price_0 + price_down) / (dS**2)
        return gamma


    def calculate_vega(self) -> float:
       
        original_sigma = self.market.sigma

        dSigma = original_sigma * self.epsilon
       
        self.market.sigma = original_sigma + dSigma
        price_up = self.mc_sim.LSM(poly_degree=2, poly_type="standard")
        
        self.market.sigma = original_sigma - dSigma
        price_down = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

        
        self.market.sigma = original_sigma

        
        vega = (price_up - price_down) / (2 * dSigma * 100)

        return vega

    def calculate_theta(self) -> float:
   
        original_maturity = self.option.maturity

        dT = original_maturity * self.epsilon

        self.option.maturity = original_maturity + dT
        price_up = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

        self.option.maturity = original_maturity - dT
        price_down = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

        self.option.maturity = original_maturity

        theta = (price_down - price_up) / (2 * dT) * 365

        return theta

    def calculate_rho(self) -> float:
        original_r = self.market.r
        dr = original_r * self.epsilon

    
        self.market.r = original_r + dr
        price_up = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

    
        self.market.r = original_r - dr
        price_down = self.mc_sim.LSM(poly_degree=2, poly_type="standard")

    
        self.market.r = original_r

        rho = (price_up - price_down) / (2 * dr)
        return rho

    
greeks_mc = Greeks(market, option, mc_sim, epsilon = 0.1)

delta_mc = greeks_mc.calculate_delta()
gamma_mc = greeks_mc.calculate_gamma()
vega_mc = greeks_mc.calculate_vega()
theta_mc = greeks_mc.calculate_theta()
rho_mc = greeks_mc.calculate_rho()

print(f"Delta : {delta_mc:.4f}")
print(f"Gamma : {gamma_mc:.4f}")
print(f"Vega : {vega_mc:.4f}")
print(f"Theta : {theta_mc:.4f}")
print(f"Rho : {rho_mc:.4f}")

