# Standard library imports
import os

# Third-party library imports
import rasterio
import numpy as np
from scipy import ndimage

"""
This script performs bicubic interpolation on a coarse-resolution GeoTIFF raster
to match the exact dimensions of a fine-resolution reference raster.
It reads paired coarse and fine TIFF files, upscales the coarse data using bicubic interpolation,
preserves full geospatial metadata from the reference fine raster, and saves the interpolated output.
Built-in statistical summaries are provided for validation of input and output data ranges.
"""


def read_tif_file(file_path):
    """Read a single-band TIFF file and return data array along with geospatial metadata"""
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        profile = src.profile
        return data, transform, crs, profile


def bicubic_interpolation(coarse_data, target_shape):
    """Upscale coarse grid data to target shape using bicubic interpolation"""
    # Calculate scaling factors
    scale_y = target_shape[0] / coarse_data.shape[0]
    scale_x = target_shape[1] / coarse_data.shape[1]

    # Apply bicubic interpolation
    interpolated_data = ndimage.zoom(coarse_data, (scale_y, scale_x), order=3)

    return interpolated_data


def save_interpolated_tif(interpolated_data, fine_profile, output_path):
    """Save interpolated raster data as a GeoTIFF file"""
    # Update profile to match interpolated data dimensions
    profile = fine_profile.copy()
    profile.update({
        'dtype': rasterio.float32,
        'height': interpolated_data.shape[0],
        'width': interpolated_data.shape[1],
        'count': 1
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write output file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(interpolated_data.astype(np.float32), 1)

    print(f"Interpolated file saved to: {output_path}")


def main():
    # File paths
    coarse_file = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\ParedNew_300mesh\50_cons\Velocity (06AUG2025 12 00 00).Terrain.MIKE_12_5_FILLED.tif"
    fine_file = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\ParedNew_300mesh\50year30mesh\Velocity (06AUG2025 12 00 00).Terrain.MIKE_12_5_FILLED.tif"

    # Output path
    output_dir = r"G:\Projects\ShouXi\python\SR-PointNet\R"
    output_filename = "Coarse_grid_bicubic_interpolated.tif"
    output_path = os.path.join(output_dir, output_filename)

    print("Step 1: Reading TIFF files...")

    # Read input files
    coarse_data, coarse_transform, coarse_crs, coarse_profile = read_tif_file(coarse_file)
    fine_data, fine_transform, fine_crs, fine_profile = read_tif_file(fine_file)

    print(f"Coarse grid shape: {coarse_data.shape}")
    print(f"Fine grid shape: {fine_data.shape}")
    print(f"Coarse grid value range: {coarse_data.min():.3f} - {coarse_data.max():.3f}")
    print(f"Fine grid value range: {fine_data.min():.3f} - {fine_data.max():.3f}")

    # Count zero-value pixels
    coarse_zeros = np.sum(coarse_data == 0)
    fine_zeros = np.sum(fine_data == 0)
    print(f"Coarse grid zero pixels: {coarse_zeros} ({coarse_zeros / coarse_data.size * 100:.2f}%)")
    print(f"Fine grid zero pixels: {fine_zeros} ({fine_zeros / fine_data.size * 100:.2f}%)")

    print("\nPerforming bicubic interpolation...")

    # Upscale coarse grid to fine grid resolution via bicubic interpolation
    coarse_interpolated = bicubic_interpolation(coarse_data, fine_data.shape)

    print(f"Interpolated coarse grid shape: {coarse_interpolated.shape}")
    print(f"Interpolated value range: {coarse_interpolated.min():.3f} - {coarse_interpolated.max():.3f}")

    # Verify that interpolated output matches target shape
    if coarse_interpolated.shape == fine_data.shape:
        print("✓ Interpolation successful: output shape matches fine grid")
    else:
        print("✗ Interpolation failed: shape mismatch")
        return

    # Save interpolated output file
    print(f"\nSaving interpolated file to: {output_path}")
    save_interpolated_tif(coarse_interpolated, fine_profile, output_path)

    # Print statistical summary for validation
    print("\nInterpolation result statistics:")
    print(
        f"Original coarse grid - min: {coarse_data.min():.6f}, max: {coarse_data.max():.6f}, mean (non-zero): {coarse_data[coarse_data != 0].mean():.6f}")
    print(
        f"Interpolated data - min: {coarse_interpolated.min():.6f}, max: {coarse_interpolated.max():.6f}, mean (non-zero): {coarse_interpolated[coarse_interpolated != 0].mean():.6f}")
    print(
        f"Target fine grid - min: {fine_data.min():.6f}, max: {fine_data.max():.6f}, mean (non-zero): {fine_data[fine_data != 0].mean():.6f}")

    print("\nStep 1 completed! Interpolated file has been saved.")


if __name__ == "__main__":
    main()