from tree import Tree

class Greeks:
    """Classes pour le calcul des sensibilités Delta et Gamma par la méthode des noeuds up/down à la racine"""

    def __init__(self, tree: Tree):
        """Constructeurs de la classe Greeks."""
        self.tree: Tree = tree
        self.node = self.tree.root  # racine de l'arbre

    def calculate_delta(self) -> float:
        """Calcul du delta à la racine par la méthode des noeuds up/down."""
        delta: float = (self.node.up_node.val_opt - self.node.down_node.val_opt) / (self.node.up_node.price - self.node.down_node.price)
        return delta

    def calculate_gamma(self) -> float:
        """Calcul du delta à la racine par la méthode des noeuds up/down."""
        delta_up: float = (self.node.up_node.val_opt - self.node.val_opt) / (self.node.up_node.price - self.node.price)
        delta_down: float = (self.node.val_opt - self.node.down_node.val_opt) / (self.node.price - self.node.down_node.price)
        gamma: float = (delta_up - delta_down) / ((self.node.up_node.price - self.node.down_node.price) / 2)

        return gamma