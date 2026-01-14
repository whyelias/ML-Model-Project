
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_blobs

# -----------------------------
# Mini-Batch K-Means (from scratch) + Visualization
# -----------------------------

# Config
np.random.seed(23)
N_SAMPLES = 500
N_FEATURES = 2
K = 3
BATCH_SIZE = 32         # mini-batch size
N_ITERS = 60            # number of mini-batch updates/frames
RANDOM_STATE = 23

# Generate synthetic dataset
X, y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=K, random_state=RANDOM_STATE)

# Initialize centers randomly in range [-2, 2]
centers = 2 * (2 * np.random.random((K, N_FEATURES)) - 1)
# Keep counts of how many points have updated each center (for 1/t learning rate)
counts = np.zeros(K, dtype=np.int64)

# History containers for visualization
centers_history = [centers.copy()]
labels_history = []

# Helper: assign labels given centers
def assign_labels(points, cents):
    # distances: (n_points, k)
    dists = np.linalg.norm(points[:, None, :] - cents[None, :, :], axis=2)
    return np.argmin(dists, axis=1)

# Run mini-batch updates and record the trajectory
for it in range(N_ITERS):
    batch_idx = np.random.choice(X.shape[0], size=BATCH_SIZE, replace=False)
    batch = X[batch_idx]

    # Assign each batch point to nearest center and update that center with learning rate 1/count
    for x in batch:
        # find nearest center
        j = np.argmin(np.linalg.norm(centers - x, axis=1))
        counts[j] += 1
        eta = 1.0 / counts[j]  # learning rate decreases as center j sees more points
        centers[j] = (1 - eta) * centers[j] + eta * x

    # Record centers after this mini-batch update
    centers_history.append(centers.copy())
    # For visualization, compute labels of ALL points with current centers
    labels_history.append(assign_labels(X, centers))

# -----------------------------
# Build animation
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_title("Mini-Batch K-Means: incremental updates")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.grid(True, alpha=0.3)

scatter = ax.scatter(X[:, 0], X[:, 1], s=25, c='lightgray')
centers_scatter = ax.scatter(centers_history[0][:, 0], centers_history[0][:, 1],
                             marker='X', s=200, c=['tab:red', 'tab:blue', 'tab:green'], edgecolor='k', linewidth=1.5,
                             label='Centers')
legend = ax.legend(loc='upper right')

# To draw center trajectories
traj_lines = [ax.plot([], [], color=c, lw=1.8, alpha=0.8)[0] for c in ['tab:red', 'tab:blue', 'tab:green']]

# Initialize function
def init():
    scatter.set_array(np.array([]))  # colors will be updated per frame
    for ln in traj_lines:
        ln.set_data([], [])
    centers_scatter.set_offsets(centers_history[0])
    return scatter, centers_scatter, *traj_lines

# Update function per frame
def update(frame):
    # frame runs from 0 .. N_ITERS-1 for labels_history; centers_history has length N_ITERS+1
    current_centers = centers_history[frame + 1]
    current_labels = labels_history[frame]

    # Recolor points based on current labels
    cmap = matplotlib.colors.ListedColormap(['#ff6961', '#779ecb', '#77dd77'])  # soft red/blue/green
    scatter.set_facecolor(cmap(current_labels))
    scatter.set_edgecolor('none')

    # Update center markers
    centers_scatter.set_offsets(current_centers)

    # Update trajectories (plot path of each center over time)
    for j in range(K):
        path = np.array([c[j] for c in centers_history[:frame+2]])  # up to current frame
        traj_lines[j].set_data(path[:, 0], path[:, 1])

    ax.set_title(f"Mini-Batch K-Means: iteration {frame+1}/{N_ITERS}")
    return scatter, centers_scatter, *traj_lines

anim = FuncAnimation(fig, update, frames=N_ITERS, init_func=init, interval=200, blit=False)

# Save animation as GIF
gif_path = "mini_batch_kmeans_animation.gif"
writer = PillowWriter(fps=5)
anim.save(gif_path, writer=writer)

# Also produce a final static figure
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.set_title("Mini-Batch K-Means: final clustering")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.grid(True, alpha=0.3)
final_centers = centers_history[-1]
final_labels = assign_labels(X, final_centers)
ax2.scatter(X[:, 0], X[:, 1], s=25, c=final_labels, cmap=matplotlib.colors.ListedColormap(['#ff6961', '#779ecb', '#77dd77']))
ax2.scatter(final_centers[:, 0], final_centers[:, 1], marker='X', s=200, c=['tab:red', 'tab:blue', 'tab:green'], edgecolor='k', linewidth=1.5, label='Centers')
ax2.legend(loc='upper right')
final_png_path = "mini_batch_kmeans_final.png"
fig2.savefig(final_png_path, bbox_inches='tight')
fig2.show()

