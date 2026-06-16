import cv2
import os
import numpy as np
from natsort import natsorted

from plotFunctions import process_all_frames_Sanjiang
"""
The process_all_frames_Sanjiang function overlays model-predicted TIF-format water depth and velocity field results
with a DEM layer, generates visualizations and saves them locally for subsequent video synthesis.
"""
if __name__ == "__main__":
    # dem_file = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\dem_30.tif"
    # pred_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\D_pred"
    # true_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\D_true"
    # out_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\comparison_frames_D"

    dem_file = r"G:\Projects\ShouXi\python\SR-PointNet\dataset\sourceData\geo\dem_sanjiang_1_5.tif"
    pred_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_Sanjiang\D_pred"
    true_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_Sanjiang\D_true"
    out_dir = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_Sanjiang\comparison_frames_Ds"

    process_all_frames_Sanjiang(dem_file, pred_dir, true_dir, out_dir,
                       video_name="pred_vs_true.mp4", fps=2)


"""
The create_video_from_images function generates a video with specified frame rate from an image sequence.
To generate a video, run the script above first, then comment out the upper code block and uncomment the code below.
"""

# def create_video_from_images(image_folder, output_video_path, fps=2):
#     """
#     Synthesize a video from a sequence of images.
#
#     Parameters:
#         image_folder: path to the folder containing input images
#         output_video_path: path to save the output video
#         fps: frames per second, default 2
#     """
#     # Retrieve all image files
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#
#     # Apply natural sort to ensure correct frame order
#     images = natsorted(images)
#
#     if not images:
#         print("No PNG image files found")
#         return
#
#     print(f"Found {len(images)} images")
#
#     # Read the first image to retrieve frame dimensions
#     first_image_path = os.path.join(image_folder, images[0])
#     frame = cv2.imread(first_image_path)
#     height, width, layers = frame.shape
#
#     # Define video codec and output parameters
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Alternatively use 'XVID'
#     video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     # Add images to the video frame by frame
#     for i, image in enumerate(images):
#         image_path = os.path.join(image_folder, image)
#         frame = cv2.imread(image_path)
#
#         if frame is None:
#             print(f"Failed to read image: {image_path}")
#             continue
#
#         # Ensure all images have consistent dimensions
#         if frame.shape[0] != height or frame.shape[1] != width:
#             print(f"Resizing image: {image}")
#             frame = cv2.resize(frame, (width, height))
#
#         video.write(frame)
#         print(f"Added frame {i + 1}/{len(images)}: {image}")
#
#     # Release the video writer object
#     video.release()
#     print(f"Video saved to: {output_video_path}")
#
#
# # Main program
# if __name__ == "__main__":
#     # Configure path parameters
#     image_folder = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\comparison_frames_U"
#     output_video_path = r"G:\Projects\ShouXi\python\SR-PointNet\resultSaving_300_geo_realRainfall\Compare_Velocity.mp4"
#     fps = 2  # Frame rate
#
#     # Generate video
#     create_video_from_images(image_folder, output_video_path, fps)
#
#     print("Video creation completed!")