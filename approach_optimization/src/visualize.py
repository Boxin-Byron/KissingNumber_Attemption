import matplotlib.pyplot as plt
import numpy as np

def plot_2d_solution(points, title="2D Kissing Number Solution"):
    """
    Plots the 2D configuration of spheres.
    points: (N, 2) array-like
    """
    if points.shape[1] != 2:
        raise ValueError("Points must be 2D for this plot function.")
        
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot central sphere (radius 1, centered at 0)
    central_circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.3, label='Central Sphere')
    ax.add_patch(central_circle)
    
    # Plot surrounding spheres (radius 1, centered at points)
    # Note: points are centers at distance 2 from origin.
    
    for i, p in enumerate(points):
        circle = plt.Circle((p[0], p[1]), 1.0, color='blue', alpha=0.5, fill=True)
        ax.add_patch(circle)
        ax.text(p[0], p[1], str(i+1), ha='center', va='center', color='white')
        
        # Draw line to center to verify distance
        ax.plot([0, p[0]], [0, p[1]], 'k--', alpha=0.2)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.show()

def plot_loss_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("Optimization Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
