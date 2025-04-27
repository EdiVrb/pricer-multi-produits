import math
from datetime import date
from MarketData import MarketData
from param import TimeParam
from Node import Node

class Tree:
    def __init__(self, mkt: MarketData, par: TimeParam):
        """ Constructeur, défintion des attributs de classe"""
        self.par: TimeParam = par                                                             # Instance de la classe TimeParam
        self.n_step: int = self.par.n_step                                                    # Nombre de pas
        self.start_date: date = self.par.start_date                                           # Date initial de l'option
        self.mat_date : date = self.par.mat_date                                              # Maturité de l'option
        self.mkt : MarketData = mkt                                                           # Instance de la classe MarketTada
        self.deltat : float = (self.mat_date - self.start_date).days / 365 / self.n_step      # espace de temps entre chaque pas
        self.alpha : float = math.exp(self.mkt.vol * math.sqrt(3 * self.deltat))              # alpha
        self.root : Node = None                                                               # Racine de l'arbre

    def BuildTree(self):
        """
        Construit l'arbre à partir de la racine.
        Cette méthode initialise la racine de l'arbre, crée les nœuds noeuds up et down et les connecte.
        Elle attribue également des probabilités de transition pour chaque nœud.
        Ensuite, elle itère sur le nombre de pas pour construire l'arbre étape par étape.
        """
        # Création de la racine au pas 0
        self.step_time : float = 0
        self.root = Node(self.mkt.price, self.mkt, self.par, self.step_time, self.alpha, self.deltat)

        # Connexion de la racine avec le noeuds Up et Down, et affectation des probabilités de réalisation
        n: Node = self.root
        n.down_node = Node(self.mkt.price / self.alpha, self.mkt, self.par,self.step_time, self.alpha, self.deltat)
        n.up_node = Node(self.mkt.price * self.alpha, self.mkt, self.par, self.step_time, self.alpha, self.deltat)
        n.proba_global = 4 / 6
        self.root.up_node.proba_global = 1 / 6
        self.root.down_node.proba_global = 1 / 6
        self.root.up_node.down_node = n
        self.root.down_node.up_node = n

        # Boucle sur les noeuds du tronc
        for i in range(1, int(self.n_step) + 1):
            # incrémentation du temps du noeud
            self.step_time += self.deltat
            n = self.build_column(n, self.step_time)

    def build_column(self, n_prec: Node, n_date: float) -> Node:
        """
        Construit une colonne de nœuds dans l'arbre pour un pas de temps donné.
        Chaque Node de la colonne apelle la méthode BuildTriplet
        Retourne : Le nœud central de la colonne construite.
        """

        # Création des nœuds centraux
        is_trunc : bool = True
        self.build_triplet(n_prec, n_prec.next_mid, is_trunc, n_date)
        n_index: Node = n_prec

        is_trunc = False
        # Création des nœuds supérieurs
        while n_index.up_node is not None:
            n_index = n_index.up_node
            self.build_triplet(n_index, n_index.down_node.next_up, is_trunc, n_date)

        # Retour sur le tronc
        n_index = n_prec
        # Création des nœuds inférieurs
        while n_index.down_node is not None:
            n_index = n_index.down_node
            self.build_triplet(n_index, n_index.up_node.next_down, is_trunc, n_date)
        # Retour sur le tronc
        n_index = n_prec
        return n_index.next_mid

    def build_triplet(self, n_prec: Node, candidate_mid: Node, is_trunc: bool, n_date: float):
        """
        Construit le triplet de nœuds (nœud intermédiaire, supérieur et inférieur)
         du pas de temps suivant à partir d'un nœud donné.
        """

        # Définition du seuil de prunning
        seuil : float = 0.000000001

        # Création / connexion du CandidateMid :
        if candidate_mid is None:  # Création du CandidateMid s'il n'existe pas
            candidate_mid:Node = Node(n_prec.forward(n_prec.price, n_prec.n_date),
                                      self.mkt, self.par, n_date, self.alpha, self.deltat)
        candidate_mid = n_prec.find_mid(candidate_mid)
        n_prec.next_mid = candidate_mid

        # Si on est sur le tronc, on construit une liaison avec le noeud précédent (backward)
        if is_trunc:
            candidate_mid.prec_node = n_prec

        # Calcul des probabilités de transitions et de la probabilité de réalisation
        self.node_calculate_proba(n_prec,seuil)

        # Cas où la probabilité globale est inférieure au seuil
        if n_prec.proba_global < seuil:
            
            # En cas de pruning sur le dernier noeuds, création des liaisons
            if n_prec.down_node is None:
                n_prec.next_mid.up_node = n_prec.up_node.next_mid
                n_prec.up_node.next_mid.down_node = n_prec.next_mid

            if n_prec.up_node is None:
                n_prec.next_mid.down_node = n_prec.down_node.next_mid
                n_prec.down_node.next_mid.up_node = n_prec.next_mid

        # Cas où la probabilité globale est supérieur au seuil
        else:
            # Création / affectation du prochain noeuds du haut
            self.LinkNode_Up(candidate_mid, n_prec, n_date)
            # Création / affectation du prochain noeuds du bas
            self.LinkNode_Down(candidate_mid, n_prec, n_date)

    def node_calculate_proba(self, n_prec : Node, seuil : float):
        if n_prec.proba_global < seuil:
            n_prec.proba_mid = 1
            n_prec.proba_down = 0
            n_prec.proba_up = 0
            n_prec.next_mid.proba_global += n_prec.proba_global * n_prec.proba_mid
        else :
            # Calcul des probabilités de transitions de n_prec et de la probabilité de réalisation du CandidateMid
            n_prec.proba_down = n_prec.pDown(n_prec.next_mid.price, n_prec.esp, n_prec.var, self.alpha)
            n_prec.proba_up = n_prec.pUp(n_prec.next_mid.price, n_prec.esp, self.alpha, n_prec.proba_down)
            n_prec.proba_mid = n_prec.pMid(n_prec.proba_up, n_prec.proba_down)
            n_prec.next_mid.proba_global += n_prec.proba_global * n_prec.proba_mid

    def LinkNode_Up(self, candidate_mid : Node, n_prec : Node, n_date : float):


        # Vérifie si le nœud supérieur n'existe pas déjà
        if candidate_mid.up_node is None:
            # Crée un nouveau nœud supérieur
            n_up : Node = Node(candidate_mid.price * self.alpha, self.mkt, self.par, n_date, self.alpha, self.deltat)
        else:
            # Utilise le nœud supérieur existant
            n_up : Node = candidate_mid.up_node

        # Établit les lien entre les nœuds
        candidate_mid.up_node = n_up
        n_up.down_node = candidate_mid
        n_prec.next_up = n_up
        n_prec.next_up.proba_global += n_prec.proba_global * n_prec.proba_up

    def LinkNode_Down(self, candidate_mid, n_prec, n_date):
        # Vérifie si le nœud supérieur n'existe pas déjà
        if candidate_mid.down_node is None:
            # Crée un nouveau nœud supérieur
            n_down : Node = Node(candidate_mid.price / self.alpha, self.mkt, self.par, n_date, self.alpha, self.deltat)
        else:
            # Utilise le nœud supérieur existant
            n_down : Node = candidate_mid.down_node

        # Établit les lien entre les nœuds
        candidate_mid.down_node = n_down
        n_down.up_node = candidate_mid
        n_prec.next_down = n_down
        n_prec.next_down.proba_global += n_prec.proba_global * n_prec.proba_down