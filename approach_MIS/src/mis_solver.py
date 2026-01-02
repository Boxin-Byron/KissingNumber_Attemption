"""Maximum Independent Set (MIS) solver.

This project uses **Gurobi** as the only supported solver.

The experiments historically exposed multiple `method` options (greedy/cvxpy
etc.). Those paths were unused in the recommended workflow and added noise.
We keep a `method` parameter in the public function for backward CLI
compatibility, but it is effectively ignored (always Gurobi).
"""

import gurobipy as gp
from gurobipy import GRB


def solve_mis(G, time_limit=300, verbose=True, method='gurobi'):
    """
    Solve Maximum Independent Set using Gurobi optimizer.
    
    Formulation:
    maximize sum(x_i) for i in V
    subject to: x_i + x_j <= 1 for all (i,j) in E
                x_i in {0, 1}
    
    Parameters:
    -----------
    G : networkx.Graph
        Conflict graph
    time_limit : float
        Time limit in seconds (default: 300)
    verbose : bool
        Print solver output (default: True)
    method : str
        Kept for backward compatibility. Ignored (always uses Gurobi).
        
    Returns:
    --------
    mis_nodes : list
        List of nodes in the maximum independent set
    mis_size : int
        Size of the MIS
    method_used : str
        Method that was actually used
    solve_time : float
        Time taken to solve
    """
    n = G.number_of_nodes()
    
    if n == 0:
        return [], 0, 'none', 0.0
    
    # Create model
    model = gp.Model("MIS")
    if not verbose:
        model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables: x[i] = 1 if node i is in the independent set
    x = model.addVars(G.nodes(), vtype=GRB.BINARY, name="x")
    
    # Objective: maximize sum of selected nodes
    model.setObjective(gp.quicksum(x[i] for i in G.nodes()), GRB.MAXIMIZE)
    
    # Constraints: at most one endpoint of each edge can be selected
    for i, j in G.edges():
        model.addConstr(x[i] + x[j] <= 1)
    
    # Solve
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        mis_nodes = [i for i in G.nodes() if x[i].X > 0.5]
        mis_size = len(mis_nodes)
        solve_time = model.Runtime
        
        if verbose:
            print(f"Gurobi MIS: {mis_size} nodes, solved in {solve_time:.2f}s")
            if model.status == GRB.TIME_LIMIT:
                print(f"  Time limit reached, gap: {model.MIPGap * 100:.2f}%")
        
        return mis_nodes, mis_size, 'gurobi', solve_time
    else:
        raise RuntimeError(f"Gurobi solver failed with status {model.status}")

