# Chargement des librairies utiles
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt

class AutocallType(Enum):
    ATHENA = "Athena"
    PHOENIX = "Phoenix"

@dataclass
class AutocallParameters:
    underlying_spot: float
    strike_percentage: float  # En pourcentage du spot
    barrier_percentage: float  # En pourcentage du spot (barrière de protection)
    coupon_rate: float  # Taux annuel du coupon
    volatility: float  # Volatilité annualisée
    risk_free_rate: float  # Taux sans risque annualisé
    dividend_yield: float  # Rendement du dividende annualisé
    maturity: float  # Maturité en années
    observation_dates: List[float]  # Dates d'observation en années
    autocall_barriers: List[float]  # Liste des barrières d'autocall (en % du spot)
    coupon_barriers: Optional[List[float]] = None  # Barrières de coupon pour Phoenix (en % du spot)
    memory_effect: bool = False  # Effet mémoire pour les coupons (Phoenix uniquement)

class AutocallSimulation:
    def __init__(self, params: AutocallParameters, nb_simulations: int = 10000, seed: int = 42):
        self.params = params
        self.nb_simulations = nb_simulations
        self.seed = seed
        np.random.seed(seed)
        
        self.strike = self.params.underlying_spot * self.params.strike_percentage
        self.barrier = self.params.underlying_spot * self.params.barrier_percentage
        
        # Si les barrières de coupon ne sont pas spécifiées, utiliser les barrières d'autocall
        if self.params.coupon_barriers is None:
            self.params.coupon_barriers = self.params.autocall_barriers
    
    def simulate_paths(self) -> np.ndarray:
        """Simuler les trajectoires de prix de l'actif sous-jacent en utilisant un mouvement brownien géométrique"""
        dt = 1/252  # Pas de temps journalier
        n_steps = int(self.params.maturity * 252) + 1
        
        drift = (self.params.risk_free_rate - self.params.dividend_yield - 0.5 * self.params.volatility**2) * dt
        diffusion = self.params.volatility * np.sqrt(dt)
        
        # Générer les innovations aléatoires
        random_innovations = np.random.normal(0, 1, size=(self.nb_simulations, n_steps))
        
        # Simuler les trajectoires
        paths = np.zeros((self.nb_simulations, n_steps))
        paths[:, 0] = self.params.underlying_spot
        
        for t in range(1, n_steps):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * random_innovations[:, t])
        
        return paths
    
    def get_observation_indices(self) -> List[int]:
        """Convertir les dates d'observation en indices de temps"""
        return [int(date * 252) for date in self.params.observation_dates]

class AutocallPricer:
    def __init__(self, autocall_type: AutocallType):
        self.autocall_type = autocall_type
    
    def price(self, params: AutocallParameters, nb_simulations: int = 10000) -> Dict:
        """Prix de l'autocall en fonction du type (Athéna ou Phoenix)"""
        simulator = AutocallSimulation(params, nb_simulations)
        paths = simulator.simulate_paths()
        
        if self.autocall_type == AutocallType.ATHENA:
            return self._price_athena(simulator, paths)
        else:  # Phoenix
            return self._price_phoenix(simulator, paths)
    
    def _price_athena(self, simulator: AutocallSimulation, paths: np.ndarray) -> Dict:
        """Pricing d'un Athéna"""
        params = simulator.params
        observation_indices = simulator.get_observation_indices()
        
        # Résultats
        redemption_dates = np.zeros(simulator.nb_simulations)
        redemption_values = np.zeros(simulator.nb_simulations)
        coupon_payments = np.zeros((simulator.nb_simulations, len(observation_indices)))
        
        # Pour chaque trajectoire
        for i in range(simulator.nb_simulations):
            path = paths[i]
            is_autocalled = False
            
            # Pour chaque date d'observation
            for j, obs_index in enumerate(observation_indices):
                # Vérifier si le produit s'autocall à cette date
                if path[obs_index] >= params.underlying_spot * params.autocall_barriers[j]:
                    redemption_dates[i] = params.observation_dates[j]
                    # Rembourser le nominal + coupon
                    coupon_payments[i, j] = params.coupon_rate * params.observation_dates[j]
                    redemption_values[i] = 1 + coupon_payments[i, j]
                    is_autocalled = True
                    break
            
            # Si pas d'autocall, vérifier la barrière à maturité
            if not is_autocalled:
                last_date_index = observation_indices[-1]
                redemption_dates[i] = params.maturity
                
                if path[last_date_index] >= simulator.barrier:
                    # Au-dessus de la barrière: nominal + coupon final
                    coupon_payments[i, -1] = params.coupon_rate * params.maturity
                    redemption_values[i] = 1 + coupon_payments[i, -1]
                else:
                    # En-dessous de la barrière: perte en capital proportionnelle
                    redemption_values[i] = path[last_date_index] / simulator.strike
        
        # Actualiser les flux
        discount_factors = np.exp(-params.risk_free_rate * redemption_dates)
        present_values = redemption_values * discount_factors
        
        # Résultats
        price = np.mean(present_values)
        
        # Statistiques sur les payoffs
        return {
            "price": price,
            "expected_maturity": np.mean(redemption_dates),
            "probability_of_autocall": np.mean(redemption_dates < params.maturity),
            "probability_of_capital_loss": np.mean((redemption_values < 1) & (redemption_dates == params.maturity)),
            "expected_return": np.mean((redemption_values - 1) / redemption_dates) if np.mean(redemption_dates) > 0 else 0,
            "min_return": np.min(redemption_values) - 1,
            "max_return": np.max(redemption_values) - 1,
        }
    
    def _price_phoenix(self, simulator: AutocallSimulation, paths: np.ndarray) -> Dict:
        """Pricing d'un Phoenix"""
        params = simulator.params
        observation_indices = simulator.get_observation_indices()
        
        # Résultats
        redemption_dates = np.zeros(simulator.nb_simulations)
        redemption_values = np.zeros(simulator.nb_simulations)
        coupon_payments = np.zeros((simulator.nb_simulations, len(observation_indices)))
        missed_coupons = np.zeros((simulator.nb_simulations, len(observation_indices)))
        memory_coupons = np.zeros((simulator.nb_simulations, len(observation_indices)))
        
        # Pour chaque trajectoire
        for i in range(simulator.nb_simulations):
            path = paths[i]
            is_autocalled = False
            memory_count = 0  # Compteur pour l'effet mémoire
            
            # Pour chaque date d'observation
            for j, obs_index in enumerate(observation_indices):
                # Vérifier si le coupon est payé
                if path[obs_index] >= params.underlying_spot * params.coupon_barriers[j]:
                    coupon_payment = params.coupon_rate * (params.observation_dates[j] / len(observation_indices))
                    coupon_payments[i, j] = coupon_payment
                    
                    # Effet mémoire si activé
                    if params.memory_effect and memory_count > 0:
                        memory_coupons[i, j] = memory_count * coupon_payment
                        memory_count = 0
                else:
                    missed_coupons[i, j] = 1
                    if params.memory_effect:
                        memory_count += 1
                
                # Vérifier si le produit s'autocall à cette date
                if path[obs_index] >= params.underlying_spot * params.autocall_barriers[j]:
                    redemption_dates[i] = params.observation_dates[j]
                    # Rembourser le nominal (les coupons sont déjà comptabilisés)
                    redemption_values[i] = 1
                    is_autocalled = True
                    break
            
            # Si pas d'autocall, vérifier la barrière à maturité
            if not is_autocalled:
                last_date_index = observation_indices[-1]
                redemption_dates[i] = params.maturity
                
                if path[last_date_index] >= simulator.barrier:
                    # Au-dessus de la barrière: nominal
                    redemption_values[i] = 1
                else:
                    # En-dessous de la barrière: perte en capital proportionnelle
                    redemption_values[i] = path[last_date_index] / simulator.strike
        
        # Ajouter les coupons et l'effet mémoire au remboursement
        total_coupons = np.sum(coupon_payments, axis=1) + np.sum(memory_coupons, axis=1)
        redemption_values += total_coupons
        
        # Actualiser les flux
        discount_factors = np.exp(-params.risk_free_rate * redemption_dates)
        present_values = redemption_values * discount_factors
        
        # Résultats
        price = np.mean(present_values)
        
        return {
            "price": price,
            "expected_maturity": np.mean(redemption_dates),
            "probability_of_autocall": np.mean(redemption_dates < params.maturity),
            "probability_of_capital_loss": np.mean((redemption_values < 1) & (redemption_dates == params.maturity)),
            "expected_return": np.mean((redemption_values - 1) / redemption_dates) if np.mean(redemption_dates) > 0 else 0,
            "min_return": np.min(redemption_values) - 1,
            "max_return": np.max(redemption_values) - 1,
            "probability_of_coupon": 1 - np.mean(missed_coupons),
            "average_coupon_payment": np.mean(total_coupons),
        }

def plot_autocall_simulation(pricer: AutocallPricer, params: AutocallParameters, nb_paths: int = 5):
    """Tracer quelques exemples de chemins pour visualiser le produit"""
    simulator = AutocallSimulation(params, nb_paths)
    paths = simulator.simulate_paths()
    observation_indices = simulator.get_observation_indices()
    
    plt.figure(figsize=(12, 8))
    
    # Tracer les chemins
    time_points = np.linspace(0, params.maturity, paths.shape[1])
    for i in range(nb_paths):
        plt.plot(time_points, paths[i], alpha=0.7)
    
    # Ajouter les niveaux importants
    plt.axhline(y=params.underlying_spot, color='k', linestyle='-', alpha=0.3, label="Niveau initial")
    plt.axhline(y=simulator.strike, color='r', linestyle='--', alpha=0.5, label="Strike")
    plt.axhline(y=simulator.barrier, color='r', linestyle=':', alpha=0.5, label="Barrière de protection")
    
    # Ajouter les barrières d'autocall
    for i, obs_date in enumerate(params.observation_dates):
        autocall_level = params.underlying_spot * params.autocall_barriers[i]
        plt.scatter(obs_date, autocall_level, color='g', marker='o', s=50)
        plt.axvline(x=obs_date, color='g', linestyle='--', alpha=0.2)
    
    plt.title(f"Simulation de {nb_paths} trajectoires pour un {pricer.autocall_type.value}")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix du sous-jacent")
    plt.legend()
    plt.grid(True)
    
    return plt

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple avec un produit Athéna (5 ans, observation annuelle)
    athena_params = AutocallParameters(
        underlying_spot=100.0,
        strike_percentage=1.0,
        barrier_percentage=0.6,
        coupon_rate=0.07,  # 7% par an
        volatility=0.2,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        maturity=5.0,
        observation_dates=[1.0, 2.0, 3.0, 4.0, 5.0],
        autocall_barriers=[1.0, 0.95, 0.90, 0.85, 0.80],
    )
    
    # Exemple avec un produit Phoenix (5 ans, observation trimestrielle)
    phoenix_params = AutocallParameters(
        underlying_spot=100.0,
        strike_percentage=1.0,
        barrier_percentage=0.6,
        coupon_rate=0.08,  # 8% par an
        volatility=0.25,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        maturity=5.0,
        observation_dates=[i/4 for i in range(1, 21)],  # Trimestriel sur 5 ans
        autocall_barriers=[1.0] * 20,  # Barrière d'autocall constante à 100%
        coupon_barriers=[0.8] * 20,  # Barrière de coupon à 80%
        memory_effect=True,  # Effet mémoire activé
    )
    
    # Créer les pricers
    athena_pricer = AutocallPricer(AutocallType.ATHENA)
    phoenix_pricer = AutocallPricer(AutocallType.PHOENIX)
    
    # Effectuer le pricing
    athena_result = athena_pricer.price(athena_params, nb_simulations=10000)
    phoenix_result = phoenix_pricer.price(phoenix_params, nb_simulations=10000)
    
    # Afficher les résultats
    print("Prix de l'Athéna:", athena_result["price"])
    print("Prix du Phoenix:", phoenix_result["price"])
    
    # Tracer des exemples de simulation
    plot_athena = plot_autocall_simulation(athena_pricer, athena_params)
    plot_phoenix = plot_autocall_simulation(phoenix_pricer, phoenix_params)
    plot_athena.show()
    plot_phoenix.show()