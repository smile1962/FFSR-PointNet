import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import BoundaryNorm
from pathlib import Path
import re

# -----------------------------
# 0) Global settings
# -----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
globalFontSize = 10

# -----------------------------
# 1) Read data
# -----------------------------
pred_path = r"E:\Project\Shouxi\paper\Fig\Yuhuo\YuhuoSourceData\D_int\2019-08-20_04_00.tif"
true_path = r"E:\Project\Shouxi\paper\Fig\Yuhuo\YuhuoSourceData\D_true\2019-08-20_04_00.tif"

with rasterio.open(pred_path) as src_pred:
    pred_data = src_pred.read(1)
    pred_nodata = src_pred.nodata

with rasterio.open(true_path) as src_true:
    true_data = src_true.read(1)
    true_nodata = src_true.nodata

mask = (
    (pred_data != pred_nodata) & (true_data != true_nodata) &
    (~np.isnan(pred_data)) & (~np.isnan(true_data)) &
    (pred_data > 0) & (true_data > 0)
)

x = pred_data[mask].ravel()
y = true_data[mask].ravel()

# -----------------------------
# 2) Automatically detect the field type and labels
# -----------------------------
def infer_field_and_source(path_str):
    """
    Detected from the path:
    - field: D / U
    - source: pred / int / true
    """
    path_str = str(path_str)
    m = re.search(r'([DU])_(pred|int|true)', path_str, re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot identify the field type from the path; check the path format: {path_str}")
    field = m.group(1).upper()   # D or U
    source = m.group(2).lower()   # pred / int / true
    return field, source

def get_plot_config(pred_path, true_path):
    field, pred_source = infer_field_and_source(pred_path)
    _, true_source = infer_field_and_source(true_path)

    # Field type
    if field == 'D':
        field_name = 'Depth'
        unit = '(m)'
        cmap = cmap_depth
    elif field == 'U':
        field_name = 'Vel. Mag.'
        unit = '(m/s)'
        cmap = cmap_vel
    else:
        raise ValueError(f"unknown field type: {field}")

    # X-axis source
    if pred_source == 'pred':
        x_source_name = 'AI'
    elif pred_source == 'int':
        x_source_name = 'Interpolation'
    elif pred_source == 'true':
        x_source_name = 'Ground Truth'
    else:
        raise ValueError(f"unknown prediction source type: {pred_source}")

    # The Y axis is normally fixed to the ground truth
    if true_source != 'true':
        raise ValueError(f"true_path should point to *_true, but was identified as: {true_source}")

    xlabel = f'{field_name} - {x_source_name} {unit}'
    ylabel = f'{field_name} - Ground Truth {unit}'
    return cmap, xlabel, ylabel

# -----------------------------
# 3) Statistics
# -----------------------------
bin_edges = [0, 0.5, 1, 2, 15]
n_bins = len(bin_edges) - 1

counts, _, _ = np.histogram2d(x, y, bins=[bin_edges, bin_edges])
percentages = (counts / len(x)) * 100

# -----------------------------
# 4) Colour configuration (separate for depth / velocity)
# -----------------------------
color_bounds = np.array([0, 1, 2, 5, 10, 20, 30, 50])
n_colors = len(color_bounds) - 1

cmap_vel = plt.get_cmap('RdBu_r', n_colors)
cmap_depth = plt.get_cmap('jet', n_colors)

# Automatically select the colormap and label for the current figure
cmap, xlabel, ylabel = get_plot_config(pred_path, true_path)
norm = BoundaryNorm(color_bounds, cmap.N)

# -----------------------------
# 5) Plotting
# -----------------------------
fig, ax = plt.subplots(figsize=(3, 3), dpi=600)
ax.set_box_aspect(1)

plot_edges = np.arange(n_bins + 1)

mesh = ax.pcolormesh(
    plot_edges, plot_edges, percentages.T,
    cmap=cmap, norm=norm,
    edgecolors='white', linewidth=1
)

# Draw the percentage text
for i in range(n_bins):
    for j in range(n_bins):
        val = percentages[i, j]
        t_color = 'white' if (val > 30 or val < 5) else 'black'
        ax.text(
            i + 0.5, j + 0.5,
            f'{val:.1f}%',
            ha='center', va='center',
            color=t_color,
            fontsize=8
        )

# Set the ticks
ax.set_xticks(plot_edges)
ax.set_yticks(plot_edges)
ax.set_xticklabels(['0', '0.5', '1', '2', '15'])
ax.set_yticklabels(['0', '0.5', '1', '2', '15'])

# Automatic labels
ax.set_xlabel(xlabel, fontsize=globalFontSize)
ax.set_ylabel(ylabel, fontsize=globalFontSize)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

ax.tick_params(direction='out', width=0.8, length=3, labelsize=globalFontSize)

plt.tight_layout(pad=0.6)

# plt.savefig('interval_U_Int.tif', format='tif', dpi=600, bbox_inches='tight')
# plt.savefig('interval_U_AI.tif', format='tif', dpi=600, bbox_inches='tight')

plt.savefig('interval_D_Int.tif', format='tif', dpi=600, bbox_inches='tight')
# plt.savefig('interval_D_AI.tif', format='tif', dpi=600, bbox_inches='tight')
plt.show()