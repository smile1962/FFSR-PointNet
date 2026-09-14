import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# -----------------------------
# 0) Global font and layout settings
# -----------------------------
# Set all Latin text to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# Set the math font to match Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'
# Global font size
globalFontSize = 10

# -----------------------------
# 1) Read the actual TIF data
# -----------------------------
# pred_path = r"E:\Project\Shouxi\paper\Fig\Yuhuo\YuhuoSourceData\D_pred\2019-08-20_04_00.tif"
# true_path = r"E:\Project\Shouxi\paper\Fig\Yuhuo\YuhuoSourceData\D_true\2019-08-20_04_00.tif"

pred_path = r"E:\Project\Shouxi\paper\Fig\Yuhuo\YuhuoSourceData\U_pred\2019-08-20_04_00.tif"
true_path = r"E:\Project\Shouxi\paper\Fig\Yuhuo\YuhuoSourceData\U_true\2019-08-20_04_00.tif"

# Use rasterio to read the raster matrix and its NoData value
with rasterio.open(pred_path) as src_pred:
    pred_data = src_pred.read(1)
    pred_nodata = src_pred.nodata

with rasterio.open(true_path) as src_true:
    true_data = src_true.read(1)
    true_nodata = src_true.nodata

# Key: because this is geographic data, NoData, NaN and cells with depth <= 0 must be removed so they do not distort the scatter and density computation
mask = (
    (pred_data != pred_nodata) & (true_data != true_nodata) &
    (~np.isnan(pred_data)) & (~np.isnan(true_data)) &
    (pred_data > 0) & (true_data > 0) # keep dry cells with depth = 0 only if your use case requires it
)

# Extract the valid 1-D arrays
x = pred_data[mask].ravel()
y = true_data[mask].ravel()

# Clip to 0-20 to match the example (if your depths exceed 20, update this and the plotting range below)
x = np.clip(x, 0, 15)
y = np.clip(y, 0, 15)

# -----------------------------
# 2) Local density estimation
# -----------------------------
bins = 70
counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 15], [0, 15]])

x_idx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
y_idx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)

density = counts[x_idx, y_idx].astype(float)
density[density < 1] = 1.0

# Normalize to the log10 scale
density_log = np.log10(density / density.max())

# Clamp uniformly to [-4, 0]
density_log = np.clip(density_log, -4, 0)

# -----------------------------
# 3) Correlation coefficient
# -----------------------------
# Check whether the data is empty to avoid errors
if len(x) > 1:
    rho = pearsonr(x, y)[0]
else:
    rho = 0.0
    print("Warning: too few valid data points to compute the correlation coefficient.")

# -----------------------------
# 4) Plotting (set to publication-grade 600 DPI)
# -----------------------------
fig, ax = plt.subplots(figsize=(3, 3), dpi=600)

# Keep the main panel as close to square as possible
ax.set_box_aspect(1)

# Colour scheme for the water-depth colorbar
# bounds = np.linspace(-4, 0, 9)
# cmap = plt.get_cmap('jet', len(bounds) - 1)
# norm = BoundaryNorm(bounds, cmap.N)

# Colour scheme for the velocity colorbar
bounds = np.linspace(-4, 0, 11)
# Use the RdBu_r colormap (reversed red-white-blue, so -4 is blue and 0 is red).
# Obtain a discrete colormap from the number of boundaries.
cmap = plt.get_cmap('RdBu_r', len(bounds) - 1)
norm = BoundaryNorm(bounds, cmap.N)

# === [End of modification] ===

# Draw the scatter
sc = ax.scatter(
    x, y,
    c=density_log,
    s=4,
    cmap=cmap,
    norm=norm,
    edgecolors='white',
    linewidths=0.25
)

# # Water depth field
# ax.set_xlim(0, 15)
# ax.set_ylim(0, 15)
# ax.set_xticks([0, 5, 10, 15])
# ax.set_yticks([0, 5, 10, 15])
#
# ax.set_xlabel('Depth - AI (m)', fontsize=globalFontSize)
# ax.set_ylabel('Depth - Ground Truth (m)', fontsize=globalFontSize)

# Velocity field
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks([0, 2, 4, 6, 8, 10])

ax.set_xlabel('Vel. Mag. - AI (m/s)', fontsize=globalFontSize)
ax.set_ylabel('Vel. Mag. - Ground Truth (m/s)', fontsize=globalFontSize)

# Thicken the axes frame
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

ax.tick_params(direction='out', width=0.8, length=3, labelsize=globalFontSize)

# -----------------------------
# 5) Inset colorbar (fixed version)
# -----------------------------
# width and height are changed to real numbers (inches) here
# 0.15 in wide and 1.5 in tall, which fits well in a 5 in wide figure
cax = inset_axes(
    ax,
    width=0.15,               # real number in inches
    height=0.8,              # real number in inches
    bbox_to_anchor=(0.15, 0.9), # position adjustment: (x, y) coordinates between 0 and 1
    bbox_transform=ax.transAxes,
    loc='upper left',
    borderpad=0
)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cb = fig.colorbar(
    sm,
    cax=cax,
    ticks=[0, -1, -2, -3, -4],
    boundaries=bounds,
    spacing='proportional'
)

# Style the colorbar ticks
cb.outline.set_linewidth(1.0)
cb.ax.tick_params(labelsize=8, width=0.8, length=3)

# Colorbar title, using Times New Roman
cax.set_title('Density (log$_{10}$)', fontsize=8, pad=8, fontfamily='Times New Roman')

# -----------------------------
# 6) Correlation annotation
# -----------------------------
ax.text(
    0.62, 0.16,
    f'ρ={rho:.4f}',
    transform=ax.transAxes,
    fontsize=globalFontSize
)

plt.tight_layout(pad=0.6)
# To save a high-resolution image directly for the manuscript, uncomment the line below
plt.savefig('scatter_U_AI.tif', format='tif', dpi=600, bbox_inches='tight')

# plt.savefig('scatter_D_AI.tif', format='tif', dpi=600, bbox_inches='tight')

plt.show()