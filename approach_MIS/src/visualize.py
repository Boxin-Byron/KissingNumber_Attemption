"""
Visualization tools for the Kissing Number problem using MIS approach.

This module provides functions to visualize:
- 2D configurations (circles)
- 3D configurations (spheres)
- Conflict graphs
- Solution quality metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_configuration(points, mis_nodes=None, title="2D Kissing Number Configuration", 
                         save_path=None, show=True):
    """
    Plot 2D configuration of unit spheres on a plane.
    
    For kissing number: spheres have radius 1, centers at distance 2 from origin.
    
    Parameters:
    -----------
    points : np.ndarray
        Array of shape (n, 2) containing all candidate points
    mis_nodes : list, optional
        Indices of points in the MIS (to highlight)
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot (default: True)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot central sphere (unit circle centered at origin, radius 1)
    central_circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.3, 
                               linewidth=2, fill=True, label='Central sphere (r=1)')
    ax.add_patch(central_circle)
    
    # Plot all candidate positions as small dots
    ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=20, 
              alpha=0.5, label='Candidates')
    
    # Plot selected spheres (MIS)
    if mis_nodes is not None:
        selected_points = points[mis_nodes]
        
        # Draw surrounding spheres (radius 1)
        for i, idx in enumerate(mis_nodes):
            p = points[idx]
            circle = plt.Circle((p[0], p[1]), 1.0, color='blue', 
                              alpha=0.5, fill=True, edgecolor='darkblue', linewidth=2)
            ax.add_patch(circle)
            
            # Label
            ax.text(p[0], p[1], str(i+1), ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=10)
            
            # Draw line to center
            ax.plot([0, p[0]], [0, p[1]], 'k--', alpha=0.3, linewidth=1)
        
        ax.scatter(selected_points[:, 0], selected_points[:, 1], 
                  c='red', s=50, marker='x', linewidths=2, 
                  label=f'MIS centers (n={len(mis_nodes)})', zorder=5)
    
    # Set aspect and limits
    max_coord = max(np.abs(points).max(), 4.0)
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    ax.set_aspect('equal')
    
    # Formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_configuration(points, mis_nodes=None, title="3D Kissing Number Configuration",
                         save_path=None, show=True, elev=20, azim=45):
    """
    Plot 3D configuration of unit spheres.
    
    For kissing number: spheres have radius 1, centers at distance 2 from origin.
    
    Parameters:
    -----------
    points : np.ndarray
        Array of shape (n, 3) containing all candidate points
    mis_nodes : list, optional
        Indices of points in the MIS (to highlight)
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot (default: True)
    elev : float
        Elevation angle for 3D view (default: 20)
    azim : float
        Azimuth angle for 3D view (default: 45)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw central sphere (radius 1)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', 
                   alpha=0.2, linewidth=0, label='Central sphere (r=1)')
    
    # Plot all candidate positions
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='lightgray', s=10, alpha=0.3, label='Candidates')
    
    # Plot selected spheres (MIS)
    if mis_nodes is not None:
        selected_points = points[mis_nodes]
        
        # Draw spheres at selected positions (radius 1, full size)
        for i, idx in enumerate(mis_nodes):
            p = points[idx]
            
            # Draw full-size sphere representations (radius = 1)
            x_s = p[0] + 1.0 * x_sphere
            y_s = p[1] + 1.0 * y_sphere
            z_s = p[2] + 1.0 * z_sphere
            ax.plot_surface(x_s, y_s, z_s, color='blue', 
                          alpha=0.3, linewidth=0)
            
            # Draw line to center
            ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 
                   'k--', alpha=0.3, linewidth=1)
        
        # Highlight centers
        ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2],
                  c='red', s=100, marker='o', edgecolors='darkred', linewidths=2,
                  label=f'MIS centers (n={len(mis_nodes)})', zorder=5)
    
    # Set limits and aspect
    max_coord = max(np.abs(points).max(), 3.0)
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    ax.set_zlim(-max_coord, max_coord)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotation instead of legend (3D legend has issues)
    if mis_nodes is not None:
        info_text = f'MIS centers: {len(mis_nodes)}\nCandidates: {len(points)}'
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_distance_histogram(points, mis_nodes=None, min_dist=2.0, 
                           title="Pairwise Distance Distribution",
                           save_path=None, show=True):
    """
    Plot histogram of pairwise distances among selected points.
    
    Parameters:
    -----------
    points : np.ndarray
        Array of shape (n, d) containing all points
    mis_nodes : list, optional
        Indices of points to analyze (if None, use all points)
    min_dist : float
        Minimum allowed distance (shown as vertical line)
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot (default: True)
    """
    from scipy.spatial.distance import pdist
    
    if mis_nodes is not None:
        selected_points = points[mis_nodes]
    else:
        selected_points = points
    
    # Compute pairwise distances
    distances = pdist(selected_points)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(min_dist, color='red', linestyle='--', linewidth=2, 
              label=f'Min distance threshold = {min_dist}')
    
    # Statistics
    min_d = np.min(distances)
    mean_d = np.mean(distances)
    
    ax.axvline(min_d, color='orange', linestyle=':', linewidth=2, 
              label=f'Actual min = {min_d:.4f}')
    
    # Formatting
    ax.set_xlabel('Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    textstr = f'n = {len(selected_points)}\n'
    textstr += f'Min: {min_d:.4f}\n'
    textstr += f'Mean: {mean_d:.4f}\n'
    textstr += f'Max: {np.max(distances):.4f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_nearest_neighbor_histogram(selected_points, min_dist=2.0,
                                   title="Nearest-Neighbor Distance Distribution",
                                   save_path=None, show=True):
    """Plot histogram of nearest-neighbor distances for a selected point set.

    This works for any dimension (2D/3D/4D/5D/...).
    """
    import numpy as _np

    if selected_points is None:
        raise ValueError("selected_points is required")

    pts = _np.asarray(selected_points, dtype=float)
    n = int(pts.shape[0])
    if n <= 1:
        nn = _np.asarray([], dtype=float)
    else:
        # O(n^2) but n is small for MIS solutions (tens).
        diff = pts[:, None, :] - pts[None, :, :]
        d = _np.linalg.norm(diff, axis=2)
        _np.fill_diagonal(d, _np.inf)
        nn = d.min(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    if nn.size > 0:
        ax.hist(nn, bins=20, color='steelblue', alpha=0.8, edgecolor='black')
        ax.axvline(min_dist, color='red', linestyle='--', linewidth=2,
                   label=f"Target min_dist={min_dist}")
        ax.axvline(nn.min(), color='black', linestyle=':', linewidth=2,
                   label=f"min nn={nn.min():.6f}")
    ax.set_xlabel('Nearest-neighbor distance')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


