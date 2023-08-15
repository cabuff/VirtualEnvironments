import numpy as np
import cv2
import argparse
from osgeo import gdal

def get_geotiff_data(geotiff_path):
    # Open the GeoTIFF file
    dataset = gdal.Open(geotiff_path)
    if dataset is None:
        raise Exception("Error opening GeoTIFF file.")
    # Get the geotransform (pixel size and position) of the GeoTIFF
    geotransform = dataset.GetGeoTransform()
    # Read the data
    img = dataset.ReadAsArray()
    band = dataset.GetRasterBand(1)
    # Calculate the area covered by one pixel
    pixel_width = abs(geotransform[1])
    pixel_height = abs(geotransform[5])
    # Get bit depth
    color_depth = gdal.GetDataTypeSize(band.DataType)
    # Calculate vertical distance covered by one color step
    min_elevation, max_elevation = band.ComputeRasterMinMax()
    color_step_dist = (max_elevation - min_elevation) / (2**color_depth - 1)
    # Close the GeoTIFF dataset
    dataset = None
    return img, color_depth, pixel_width, pixel_height, color_step_dist


def apply_gaussian_blur(img, kernel_size, sigma_x):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x)
    return blurred_image


def calculate_resize_factors(orig_pixel_width, orig_pixel_height, new_pixel_size):
    resize_factor_x = orig_pixel_width / new_pixel_size
    resize_factor_y = orig_pixel_height / new_pixel_size
    return resize_factor_x, resize_factor_y


def resize_image(img, resize_factor_x, resize_factor_y):
    # Get the original width and height of the image
    orig_height, orig_width = img.shape[:2]
    # Calculate the new width and height using the resize factor
    new_width = int(orig_width * resize_factor_x)
    new_height = int(orig_height * resize_factor_y)
    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image


def normalize_to_uint16(img, new_color_step_dist, use_binary_mode):
    # Normalize the data to fit within the 16-bit range (0 to 65535)
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    if use_binary_mode:
        normalized_img = np.rint(normalized_img)
    if not use_binary_mode and new_color_step_dist > 0 and new_color_step_dist < 65535:
        color_range = (img_max - img_min) / new_color_step_dist
    else:
        color_range = 65535
    normalized_img *= color_range
    normalized_img = (np.clip(normalized_img, 0, color_range) + ((65535 - color_range) / 2)).astype(np.uint16)
    color_step_dist = (img_max - img_min) / color_range
    return normalized_img, color_step_dist


def show_info(path, color_depth, pixel_width, pixel_height, color_step_dist, use_binary_mode):
    print("File path:", path)
    print("Color depth:", color_depth, "bit")
    print("Pixel width:", pixel_width, "units")
    print("Pixel height:", pixel_height, "units")
    if not use_binary_mode:
        print("Distance covered by one color step:", color_step_dist, "units")


def main(geotiff_path, new_pixel_size, new_color_step_dist, add_blur, use_binary_mode):
    output_path = '{}.png'.format(geotiff_path.rsplit('.', 1)[0])

    img, color_depth, pixel_width, pixel_height, color_step_dist = get_geotiff_data(geotiff_path)
    print("INPUT")
    show_info(geotiff_path, color_depth, pixel_width, pixel_height, color_step_dist, use_binary_mode)

    print("Processing...")
    if new_pixel_size:
        new_pixel_size = float(new_pixel_size)
        resize_factor_x, resize_factor_y = calculate_resize_factors(pixel_width, pixel_height, new_pixel_size)
        img = resize_image(img, resize_factor_x, resize_factor_y)
    if add_blur:
        print("GAUSSIAN BLUR SETTINGS")
        blur_kernel_size = int(input("Enter kernel size (odd number): "))
        blur_intensity = int(input("Enter blur intensity (sigma x): "))
        img = apply_gaussian_blur(img, blur_kernel_size, blur_intensity)
    new_color_step_dist = float(new_color_step_dist)
    img, output_color_step_dist = normalize_to_uint16(img, new_color_step_dist, use_binary_mode)
    print("Done")

    print("RESULT")
    show_info(output_path, 16, new_pixel_size if new_pixel_size else pixel_width, new_pixel_size if new_pixel_size else pixel_height, output_color_step_dist, use_binary_mode)

    # Save result
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert/Adjust GeoTIFF files")
    parser.add_argument("input_file", help="Path to GeoTIFF image file (.tif)")
    parser.add_argument("-s", "--new_pixel_size", help="Width and height that should be covered by one pixel in the output image", default=0)
    parser.add_argument("-c", "--new_color_step_dist", help="Distance that should be covered vertically by one color step in the output image", default=0)
    parser.add_argument("-g", "--gblur", action="store_true", help="Add gaussian blur")
    parser.add_argument("-b", "--binary", action="store_true", help="Enable binary mode")
    args = parser.parse_args()
    main(args.input_file, args.new_pixel_size, args.new_color_step_dist, args.gblur, args.binary)
