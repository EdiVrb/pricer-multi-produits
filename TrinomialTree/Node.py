import math
from prompt_toolkit.filters import emacs_selection_mode
from MarketData import MarketData
from OptTrade import OptTrade2
from param import TimeParam
from MarketData import MarketData

class Node:
    """ Classe permettant de définir les noeuds, les attributs associés (prix du sous jacent, date, probabilités, prix de l'option ...)
    à chaque noeud ainsi que les branchements avec les autres noeuds
    """
    def __init__(self, price: float, mkt: MarketData, par: TimeParam, n_date: float, alpha:float, deltat:float):
        # Initialisation des attributs principaux
        self.par : TimeParam = par          # intance de la classe TimeParam
        self.mkt: MarketData = mkt          # instance de la classe MarketData
        self.alpha: float = alpha           # alpha
        self.deltat: float = deltat         # temps associé à un pas
        self.n_date: float = n_date         # date du  noeud
        self.price: float = price           # prix du sous jacent au noeud
        self.VI: float = 0                  # Pay-Off de l'option pour un noeud
        self.val_opt: float = 0             # prix de l'option pour un noeud

        # Calcul de l'espérance et de la variance
        self.esp : float = self.forward(self.price, n_date)
        self.var : float = (self.price ** 2) * (math.exp(2 * self.mkt.rf * self.deltat)) * (
                                math.exp((self.mkt.vol ** 2) * self.deltat) - 1)

        # Affectation des probabilités des noeuds suivants (up, mid, down) lorsqu'on est à maturité
        if self.n_date == self.par.mat_date:
            # Récupération des probabilités
            self.proba_down: float = self.pDown(self.esp, self.esp, self.var, self.alpha)
            self.proba_up: float = self.pUp(self.esp, self.esp, self.alpha, self.proba_down)
            self.proba_mid: float = self.pMid(self.proba_up, self.proba_down)

        # Facteur d'actualisation
        self.DF: float = math.exp(-self.mkt.rf * self.deltat)
        self.is_calculated: bool = False

        # définition de liens vers les autres nœuds / branches par défaut :
        self.next_up: Node = None
        self.next_mid: Node = None
        self.next_down: Node = None
        self.up_node: Node = None
        self.down_node: Node = None
        self.prec_node: Node = None
        # Probabilité de réalisation d'un noeud par défaut
        self.proba_global: float = 0

    def pDown(self, next_price: float, esp: float, var: float, alpha: float) -> float:
        """Calcul la probabilité de descente du prix."""
        return ((next_price ** (-2) * (var + (esp ** 2))) - 1 - (alpha + 1) * (next_price ** (-1) * esp - 1)) / (
                    (1 - alpha) * (alpha ** (-2) - 1))

    def pUp(self, next_price: float, esp: float, alpha: float, pDown: float) -> float:
        """Calcul la probabilité montante du prix."""
        return (next_price ** (-1) * esp - 1 - ((alpha ** (-1) - 1) * pDown)) / (alpha - 1)

    @staticmethod
    def pMid(pUp: float, pDown: float) -> float:
        """Calcul la probabilité de rester au milieu."""
        return 1 - pUp - pDown

    # Méthode
    def forward(self, S: float, prec_date: float) -> float:
        """Calcul l'espérance du prix forward en tenant compte des dividendes éventuels."""
        # Cas où il y a une tombée de dividende pendant l'intervalle de temps entre le noeud et le noeud suivant
        if (prec_date < ((self.mkt.div_date - self.par.start_date).days / 365)) and (
            ((self.mkt.div_date - self.par.start_date).days / 365) <= prec_date + self.deltat):
            return S * math.exp(self.mkt.rf * self.deltat) - self.mkt.div
        else:
            return S * math.exp(self.mkt.rf * self.deltat)

    def find_mid(self, n_next: 'Node') -> 'Node':
        """Recherche et liaison des noeuds intermédiaires."""
        up_bound: float = self.b_sup(n_next)    # Borne supérieur de l'intervalle du forward
        low_bound: float = self.b_inf(n_next)   # Borne inférieur de l'intervalle du forward

        # Calcul du prix forward.
        fwd: float = self.forward(self.price, self.n_date)

        if fwd < low_bound:
            # Recherche du noeud du bas tant que le prix forward n'appartient pas à l'intervalle définit:
            while low_bound > fwd and fwd < up_bound:
                if n_next.down_node is None:
                    n_down: Node = Node(n_next.price / self.alpha, self.mkt, self.par, n_next.n_date, self.alpha, self.deltat)
                else:
                    n_down: Node = n_next.down_node
                # Création des connexions supérieur et inférieur
                n_down.up_node = n_next
                n_next.down_node = n_down
                n_next = n_next.down_node


                # Calcul des nouvelles bornes sup et inf
                up_bound = self.b_sup(n_next)
                low_bound = self.b_inf(n_next)

        elif up_bound < fwd:
            # Recherche du noeud du haut tant que le prix forward n'appartient pas à l'intervalle définit:
            while low_bound > fwd and fwd < up_bound:
                if n_next.up_node is None:
                    n_up: Node = Node(n_next.price * self.alpha, self.mkt, self.par, n_next.ndate, self.alpha, self.deltat)
                else:
                    n_up: Node = n_next.up_node
                    # Création des connexions supérieur et inférieur
                n_up.down_node = n_next
                n_next.up_node = n_up
                n_next = n_next.up_node

                # Calcul des nouvelles bornes sup et inf
                up_bound = self.b_sup(n_next)
                low_bound = self.b_inf(n_next)

        return n_next

    def b_sup(self, n_next: float) -> float:
        """Méthode permettant de calculer la borne sup pour la recherche du forward"""
        return n_next.price * (1 + self.alpha) / 2

    def b_inf(self, n_next: float) -> float:
        """ Méthode permettant de calculer la borne inf pour la recherche du forward"""
        return n_next.price * (1 + (1 / self.alpha)) / 2

    def compute_opt_price(self, opt_trade: OptTrade2) -> float:
        """ Méthode permettant de calculer le prix de l'option pour un noeud """

        #Cas où n correspond au noeuds à maturité
        if self.next_mid is None:
            return opt_trade.VI(self.price)

        else:
            # Calcul des composantes du prix
            val_mid: float = self.next_mid.val_opt
            if not(self.next_up is None):
                val_up: float = self.next_up.val_opt
            else:
                val_up: float = 0

            if not(self.next_down is None):
                val_down: float = self.next_down.val_opt
            else:
                val_down: float = 0

            # Calcul de la valeur actualisée de l'option au noeud n
            result: float = self.DF*(val_up*self.proba_up + val_mid * self.proba_mid + val_down * self.proba_down)

            # Cas de l'exercice Américain et Européen
            if opt_trade.opt_exe == "American":
                return max(result, opt_trade.VI(self.price))
            else:
                return result

    def option_price_backard(self, opt_trade: OptTrade2) -> float:
        """Méthode pour calculer le prix de l'option en utilisant la méthode backwerd"""

        # Récupération de la racine
        n_prec: Node = self.next_mid.prec_node

        # boucle pour trouver le dernier noeuds
        while n_prec.next_mid is not None:
            n_prec = n_prec.next_mid

        # boucle backward
        while n_prec is not None:
            # Calcul du prix de l option sur le tronc
            n_prec.val_opt = n_prec.compute_opt_price(opt_trade)

            # Affectation de nIndex à la valeur située au dessus du tronc et boucle sur les noeuds du haut
            n_index: Node = n_prec.up_node
            while n_index is not None:
                # Calcul du prix de l option
                n_index.val_opt = n_index.compute_opt_price(opt_trade)
                # Affectation au prochain noeuds supérieur
                n_index  = n_index.up_node

            # Affectation de nIndex à la valeur située au dessous du tronc et boucle sur les noeuds du bas
            n_index = n_prec.down_node
            while n_index is not None:
                # Calcul du prix de l option
                n_index.val_opt = n_index.compute_opt_price(opt_trade)
                # Affectation au prochain noeuds inférieur
                n_index = n_index.down_node

            # Si on est à la racine, récupération du prix
            if n_prec.prec_node is None:
                return n_prec.val_opt
            else:
                n_prec = n_prec.prec_node