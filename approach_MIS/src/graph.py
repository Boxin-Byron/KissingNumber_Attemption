"""Conflict graph construction.

We build an undirected graph where vertices are candidate sphere centers and
edges represent geometric conflicts (overlapping spheres).

Implementation note: we always use SciPy's `cKDTree.query_pairs` to generate
conflict edges efficiently. (SciPy is a required dependency for this project.)
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def build_conflict_graph(points, min_dist=2.0, epsilon=1e-4):
    """
    Build a conflict graph from sphere center candidates.
    
    Two spheres overlap if the distance between their centers is less than min_dist.
    
    For the Kissing Number problem:
    - All spheres have radius r
    - Central sphere at origin with radius r
    - Surrounding sphere centers at distance 2r from origin
    - Two surrounding spheres overlap if ||v_i - v_j|| < 2r
    
    Default: r=1 (unit radius), so min_dist=2.0
    
    Parameters:
    -----------
    points : np.ndarray
        Array of shape (n, dim) containing candidate sphere centers
    min_dist : float, optional
        Minimum allowed distance between sphere centers (default: 2.0 for r=1)
    epsilon : float, optional
        Numerical tolerance for distance comparison (default: 1e-6)

    Returns:
    --------
    G : networkx.Graph
        Conflict graph where edges represent overlapping spheres
        Node attributes include 'pos' (position vector)
        
    Notes:
    ------
    - For unit radius spheres (r=1), min_dist = 2.0
    - An edge (i, j) exists if ||v_i - v_j|| < min_dist - epsilon
    """
    n = len(points)
    G = nx.Graph()
    
    # Add nodes with position attributes
    for i in range(n):
        G.add_node(i, pos=points[i])
    
    radius = float(min_dist) - float(epsilon)
    if radius <= 0:
        raise ValueError(f"min_dist - epsilon must be > 0, got {min_dist} - {epsilon}")

    edge_count = 0

    def _add_edges_from_pairs(pairs):
        nonlocal edge_count
        for i, j in pairs:
            G.add_edge(int(i), int(j))
            edge_count += 1

    tree = cKDTree(np.asarray(points, dtype=np.float64))
    pairs = tree.query_pairs(r=radius, output_type='set')
    _add_edges_from_pairs(pairs)
    
    print(f"Built conflict graph (kdtree): {n} nodes, {edge_count} edges")
    print(f"Graph density: {nx.density(G):.6f}")
    
    return G


def analyze_conflict_graph(G):
    """
    Analyze properties of the conflict graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        Conflict graph
        
    Returns:
    --------
    stats : dict
        Dictionary containing graph statistics
    """
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': (nx.is_connected(G) if G.number_of_nodes() > 0 else True),
        'n_components': (nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0),
    }
    
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)
        
        # Largest component
        if stats['n_components'] > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_component_size'] = len(largest_cc)
    
    return stats


def greedy_coloring_bound(G):
    """
    Compute a lower bound for the independence number using greedy coloring.
    
    The independence number α(G) >= n / χ(G), where χ(G) is the chromatic number.
    We approximate χ(G) using greedy coloring.
    
    Parameters:
    -----------
    G : networkx.Graph
        Conflict graph
        
    Returns:
    --------
    lower_bound : int
        Lower bound for the independence number
    n_colors : int
        Number of colors used in greedy coloring
    """
    if G.number_of_nodes() == 0:
        return 0, 0
    
    # Greedy coloring gives an upper bound for chromatic number
    coloring = nx.greedy_color(G, strategy='largest_first')
    n_colors = max(coloring.values()) + 1
    
    # Lower bound for independence number
    lower_bound = G.number_of_nodes() // n_colors
    
    return lower_bound, n_colors


def visualize_graph_2d(G, title="Conflict Graph"):
    """
    Visualize a 2D conflict graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        Conflict graph with 2D 'pos' node attributes
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    
    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')
    
    if not pos or len(list(pos.values())[0]) != 2:
        print("Warning: Graph nodes must have 2D 'pos' attributes for 2D visualization")
        return
    
    # Convert to dict format for networkx
    pos_dict = {node: pos[node] for node in G.nodes()}
    
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos_dict, node_color='lightblue', 
                          node_size=100, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos_dict, alpha=0.3, edge_color='red')
    
    # Draw central circle
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, 
                       linestyle='--', linewidth=2, label='Central sphere')
    plt.gca().add_patch(circle)
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
