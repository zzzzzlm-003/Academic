"""
Assignment 3, Question 2: Feasible Region and Branch-and-Bound Visualization
- 2A: Plot feasible region of the integer program
- 2B: Plot feasible region changes at key B&B nodes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Problem: max z = 4*x1 - x2
# s.t. 7*x1 - 2*x2 <= 14, x2 <= 3, 2*x1 - 2*x2 <= 3, x1, x2 in Z+

# Vertices of LP feasible region (in order for polygon, counterclockwise)
# (2,0) is NOT feasible: 2*2-0=4 > 3. Correct: (1.5,0) from 2x1-2x2=3, x2=0
# Polygon has 5 vertices: (0,0), (1.5,0), (11/5,7/10), (20/7,3), (0,3)
VERTICES = np.array([
    [0, 0],
    [1.5, 0],         # 2x1-2x2=3, x2=0
    [11/5, 7/10],     # 7x1-2x2=14, 2x1-2x2=3
    [20/7, 3],        # 7x1-2x2=14, x2=3
    [0, 3],           # x2=3, x1=0
])


def get_integer_feasible_points():
    """Integer points inside or on the LP polygon."""
    points = []
    for x1 in range(4):  # x1 up to ~3
        for x2 in range(4):  # x2 up to 3
            if _is_feasible(x1, x2):
                points.append([x1, x2])
    return np.array(points)


def _is_feasible(x1, x2):
    """Check if (x1,x2) is feasible (inside or on the LP region)."""
    if x1 < 0 or x2 < 0:
        return False
    if 7*x1 - 2*x2 > 14 + 1e-9:
        return False
    if x2 > 3 + 1e-9:
        return False
    if 2*x1 - 2*x2 > 3 + 1e-9:
        return False
    return True


def plot_2a_feasible_region(save_path=None, show=True):
    """2A: Draw feasible region of the integer program."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Shade LP feasible region
    poly = Polygon(VERTICES, alpha=0.3, facecolor='steelblue', edgecolor='navy', linewidth=2)
    ax.add_patch(poly)

    # Plot constraint lines (extended)
    x1_vals = np.linspace(-0.5, 4, 200)
    # 7x1 - 2x2 = 14  =>  x2 = (7x1 - 14)/2
    x2_c1 = (7*x1_vals - 14) / 2
    ax.plot(x1_vals, x2_c1, 'b-', lw=1.5, label=r'$7x_1 - 2x_2 = 14$')
    # x2 = 3
    ax.axhline(3, color='green', lw=1.5, label=r'$x_2 = 3$')
    # 2x1 - 2x2 = 3  =>  x2 = x1 - 1.5
    x2_c3 = x1_vals - 1.5
    ax.plot(x1_vals, x2_c3, 'purple', lw=1.5, label=r'$2x_1 - 2x_2 = 3$')

    # Integer feasible points
    int_pts = get_integer_feasible_points()
    ax.scatter(int_pts[:, 0], int_pts[:, 1], c='red', s=80, zorder=5, marker='o', edgecolors='darkred', linewidths=2)
    for pt in int_pts:
        ax.annotate(f'({int(pt[0])},{int(pt[1])})', (pt[0], pt[1]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Vertices
    ax.scatter(VERTICES[:, 0], VERTICES[:, 1], c='orange', s=100, zorder=6, marker='s', edgecolors='black', linewidths=2)

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(r'2A: Feasible Region of IP (max $z = 4x_1 - x_2$)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_2b_node_feasible_regions(save_dir=None, show=True):
    """2B: Plot feasible region changes at Node 0, 1, 3, 4."""
    if save_dir is None:
        save_dir = '.'

    # Node 0: Original LP polygon
    fig0, ax0 = plt.subplots(figsize=(6, 5))
    poly0 = Polygon(VERTICES, alpha=0.4, facecolor='steelblue', edgecolor='navy', linewidth=2)
    ax0.add_patch(poly0)
    ax0.scatter(20/7, 3, c='red', s=120, zorder=6, marker='*')
    ax0.annotate(r'LP opt: ($\frac{20}{7}$, 3), z=$\frac{59}{7}$', (20/7, 3), xytext=(15, 15), textcoords='offset points', fontsize=10)
    _add_constraint_lines(ax0)
    ax0.set_xlim(-0.5, 4)
    ax0.set_ylim(-0.5, 3.8)
    ax0.set_xlabel(r'$x_1$')
    ax0.set_ylabel(r'$x_2$')
    ax0.set_aspect('equal')
    ax0.grid(True, alpha=0.3)
    ax0.set_title('Node 0: LP Relaxation')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/q2_node0.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # Node 1: Add x1 <= 2 (left branch)
    # Feasible region: clip by x1<=2 -> (0,0), (1.5,0), (2, 0.5), (2, 3), (0, 3)
    vert1 = np.array([[0, 0], [1.5, 0], [2, 1/2], [2, 3], [0, 3]])
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    poly1 = Polygon(vert1, alpha=0.4, facecolor='steelblue', edgecolor='navy', linewidth=2)
    ax1.add_patch(poly1)
    ax1.axvline(2, color='orange', linestyle='--', lw=2, label=r'$x_1 \leq 2$')
    ax1.scatter(2, 1/2, c='red', s=120, zorder=6, marker='*')
    ax1.annotate(r'LP opt: (2, 1/2), z=15/2', (2, 0.5), xytext=(10, 10), textcoords='offset points', fontsize=10)
    _add_constraint_lines(ax1)
    ax1.set_xlim(-0.5, 4)
    ax1.set_ylim(-0.5, 3.8)
    ax1.set_xlabel(r'$x_1$')
    ax1.set_ylabel(r'$x_2$')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Node 1: Add $x_1 \\leq 2$')
    ax1.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/q2_node1.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # Node 3: Add x1<=2, x2>=1 (right of Node 1, integer feasible)
    vert3 = np.array([[2, 1], [2, 1.5], [11/5, 7/10]])
    # Actually with x2>=1 the feasible region: x1<=2, x2>=1, and inside original
    # The region is a small polygon. At x1=2, x2 from 1 to 3; at boundary 7x1-2x2=14, x2 from 1 to 3
    # Simpler: the region is bounded by x1=2, x2=1, 7x1-2x2=14, x2=3, 2x1-2x2=3
    # Vertices: (2,1), (2,3) intersection with x1=2 and... 7*2-2*x2=14 -> x2=0, so (2,0) not in x2>=1
    # x2=1 and 7x1-2=14 -> x1=16/7, but x1<=2 so (2,1). x2=1 and 2x1-2=3 -> x1=2.5, x1<=2 so (2,1)
    vert3 = np.array([[11/5, 7/10], [2, 1.5], [2, 1], [1.5, 1]])  # approximate
    # More careful: feasible = { (x1,x2): 7x1-2x2<=14, x2<=3, 2x1-2x2<=3, x1<=2, x2>=1, x1,x2>=0 }
    # At (2,1): all satisfied. The polygon vertices...
    vert3 = np.array([[1.5, 1], [2, 1], [2, 7/10 + 0.001]])  # small triangle around (2,1)
    # Actually (1.5,1) from 2x1-2=3 -> x1=2.5, so 2x1-2x2=3 gives x2=x1-1.5. At x2=1, x1=2.5 > 2. So boundary from x1=2: (2, 1) to (2, 0.5) is x2 on 2*2-2x2=3 -> x2=0.5. So (2,0.5) to (2,1) - but we need x2>=1. So we only have the line from (2,1) upward. 7*2-2x2=14 -> x2=0. So with x1=2, x2 from 0.5 to 0. So for x2>=1, we need 7x1-2*1<=14 -> x1<=16/7. And 2x1-2<=3 -> x1<=2.5. So x1<=2. So the region is: x1 in [?, 2], x2 in [1, ?]. Lower x1: 2x1-2*1=3 -> x1=2.5. So x1 can go down. For min x1 with x2=1: 7x1-2<=14 -> x1<=16/7. So vertices: (16/7, 1), (2, 1). And (2, 3) with x2=3? 7*2-6=8<=14 ok. So (2,3). And (11/5, 7/10) has x2<1 so not in. So vertices: (16/7, 1), (2, 1), (2, 3)? But (2,3) and 2*2-6=-2<=3 ok. So polygon: (16/7, 1), (2, 1), (2, 3). And (20/7, 3) has x1>2 so no. So the polygon is triangle (16/7, 1), (2, 1), (2, 3).
    vert3 = np.array([[16/7, 1], [2, 1], [2, 3]])
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    poly3 = Polygon(vert3, alpha=0.4, facecolor='green', edgecolor='darkgreen', linewidth=2)
    ax3.add_patch(poly3)
    ax3.axvline(2, color='orange', linestyle='--', lw=1.5)
    ax3.axhline(1, color='orange', linestyle='--', lw=1.5, label=r'$x_2 \geq 1$')
    ax3.scatter(2, 1, c='red', s=120, zorder=6, marker='*')
    ax3.annotate(r'IP opt: (2, 1), z=7', (2, 1), xytext=(10, 10), textcoords='offset points', fontsize=10)
    _add_constraint_lines(ax3)
    ax3.set_xlim(-0.5, 4)
    ax3.set_ylim(-0.5, 3.8)
    ax3.set_xlabel(r'$x_1$')
    ax3.set_ylabel(r'$x_2$')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Node 3: Add $x_1 \\leq 2$, $x_2 \\geq 1$ (integer feasible)')
    ax3.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/q2_node3.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # Node 4: Add x1<=2, x2<=0 (left of Node 1)
    # x2=0, so we get the segment on x2=0
    vert4 = np.array([[0, 0], [1.5, 0], [2, 0]])
    # Actually the feasible region with x2<=0 and x2>=0 gives x2=0. So we have 7x1<=14, 2x1<=3 -> x1<=1.5. So segment from (0,0) to (1.5, 0).
    vert4 = np.array([[0, 0], [1.5, 0]])
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    ax4.plot([0, 1.5], [0, 0], 'b-', lw=3, label='Feasible (x2=0)')
    ax4.axvline(2, color='orange', linestyle='--', lw=1.5)
    ax4.axhline(0, color='gray', linestyle='-', lw=1)
    ax4.scatter(1.5, 0, c='red', s=120, zorder=6, marker='*')
    ax4.annotate(r'LP opt: (3/2, 0), z=6', (1.5, 0), xytext=(10, 10), textcoords='offset points', fontsize=10)
    _add_constraint_lines(ax4)
    ax4.set_xlim(-0.5, 4)
    ax4.set_ylim(-0.5, 3.8)
    ax4.set_xlabel(r'$x_1$')
    ax4.set_ylabel(r'$x_2$')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Node 4: Add $x_1 \\leq 2$, $x_2 \\leq 0$')
    ax4.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(f'{save_dir}/q2_node4.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def _add_constraint_lines(ax):
    """Add constraint lines to an axis."""
    x1_vals = np.linspace(-0.5, 4, 100)
    x2_c1 = (7*x1_vals - 14) / 2
    ax.plot(x1_vals, x2_c1, 'b-', lw=1, alpha=0.7)
    ax.axhline(3, color='green', lw=1, alpha=0.7)
    x2_c3 = x1_vals - 1.5
    ax.plot(x1_vals, x2_c3, 'purple', lw=1, alpha=0.7)


def plot_bb_tree(save_path=None, show=True):
    """Draw the Branch-and-Bound tree as a text/matplotlib diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Node positions: right branch = higher (x2>=1), left branch = lower (x2<=0)
    # Per rule: first left (x1<=2), then right-most & prefer higher
    nodes = {
        '0': (6, 9),
        '1': (3, 6),
        '2': (9, 6),
        '3': (4, 3),   # right of Node 1: x2>=1 (higher) - explored first
        '4': (2, 3),   # left of Node 1: x2<=0 (lower)
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        ax.add_patch(plt.Circle((x, y), 0.35, fill=True, facecolor='lightblue', edgecolor='black', lw=1.5))
        ax.text(x, y, name, ha='center', va='center', fontsize=12, fontweight='bold')

    # Node labels (below each node)
    labels = {
        '0': r'$(20/7, 3)$, $z=59/7$',
        '1': r'$(2, 1/2)$, $z=15/2$',
        '2': r'Infeasible',
        '3': r'$(2, 1)$, $z=7$ ✓',
        '4': r'$(3/2, 0)$, $z=6$',
    }
    for name, (x, y) in nodes.items():
        ax.text(x, y - 0.7, labels[name], ha='center', va='top', fontsize=9)

    # Edges with branch labels
    def draw_edge(x1, y1, x2, y2, label, mid_off=0):
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5)
        mx, my = (x1+x2)/2 + mid_off, (y1+y2)/2
        ax.text(mx, my, label, fontsize=8, ha='center', va='bottom')

    draw_edge(6, 9, 3, 6, r'$x_1 \leq 2$', -0.3)
    draw_edge(6, 9, 9, 6, r'$x_1 \geq 3$', 0.3)
    # Left of Node 1: x2<=0 -> Node 4; Right: x2>=1 -> Node 3 (right-most, higher)
    draw_edge(3, 6, 2, 3, r'$x_2 \leq 0$', -0.2)
    draw_edge(3, 6, 4, 3, r'$x_2 \geq 1$', 0.2)

    # Pruning annotations
    ax.text(9, 5.5, 'Prune', fontsize=9, color='red', fontweight='bold')
    ax.text(4, 2, 'Prune\n(integer)', fontsize=8, color='red', ha='center')
    ax.text(2, 2, 'Prune\n(z=4<7)', fontsize=8, color='red', ha='center')

    ax.set_title('Branch-and-Bound Tree (Node selection: first left, then right & higher)', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


if __name__ == '__main__':
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(base, 'q2_figures')
    os.makedirs(figdir, exist_ok=True)

    print('Generating 2A: Feasible region plot...')
    plot_2a_feasible_region(save_path=os.path.join(figdir, 'q2a_feasible_region.png'), show=False)

    print('Generating 2B: B&B tree...')
    plot_bb_tree(save_path=os.path.join(figdir, 'q2b_bb_tree.png'), show=False)

    print('Generating 2B: Feasible region changes at nodes...')
    plot_2b_node_feasible_regions(save_dir=figdir, show=False)

    print(f'Done. Figures saved to {figdir}/')
