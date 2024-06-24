import numpy as np
import random
import cv2
from enum import Enum
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, scale
from descartes import PolygonPatch

class GenerationMode(Enum):
    SHADE = 0
    SCATTERED = 1

class Ellipse:
    def __init__(self, center, radius, angle):
        self.center = center
        self.radius = radius
        self.angle = angle

def check_overlap(new_ellipse, existing_ellipses):
    for ellipse in existing_ellipses:
        ellipse_1 = rotate(scale(Point(ellipse.center).buffer(1), ellipse.radius[0], ellipse.radius[1]), ellipse.angle, origin='centroid')
        ellipse_2 = rotate(scale(Point(new_ellipse.center).buffer(1), new_ellipse.radius[0], new_ellipse.radius[1]), new_ellipse.angle, origin='centroid')

        if ellipse_1.intersects(ellipse_2):
            return True

    return False

def rotate_vector(vector, angle):
    x, y = vector
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    rotated_x = x * cos_theta - y * sin_theta
    rotated_y = x * sin_theta + y * cos_theta
    
    return (rotated_x, rotated_y)

def translate(vector, translation):
    return tuple(v + t for v, t in zip(vector, translation))

def rotate_vector_around_center(vector, angle, center):
    translated_vector = translate(vector, (-center[0], -center[1]))
    rotated_vector = rotate_vector(translated_vector, angle)
    translated_back_vector = translate(rotated_vector, center)
    return translated_back_vector

def get_sample_weed_positions(image, number_of_species, number_of_samples):
    points = []
    groups = []
    i = 0

    for i in range(number_of_species):
        n_samples = number_of_samples[i]
        X = np.random.randint(0, image.shape[1], n_samples)
        Y = np.random.randint(0, image.shape[0], n_samples)

        points = [(X[i], Y[i]) for i in range(n_samples)]
        groups.append(points)

    return groups

def get_shadow_weed_positions(image, number_of_shadows, density, std_deviation, min_radius, max_radius):
    h, w = image.shape[:2]
    ellipses = []
    points_inside_ellipses = []
    i = 0

    while i < number_of_shadows:
        shadow_coordinate = (np.random.randint(0, w), np.random.randint(0, h))
        shadow_radius = (np.random.randint(min_radius*w, max_radius*w), np.random.randint(min_radius*w, max_radius*w))
        angle = np.random.randint(-90, 90)
        new_ellipse = Ellipse(shadow_coordinate, shadow_radius, angle)

        if not check_overlap(new_ellipse, ellipses):
            ellipse_area = np.pi * shadow_radius[0] * shadow_radius[1]
            num_points = int(ellipse_area / (100_000 / density))
            points = []

            X = np.random.normal(loc=shadow_coordinate[0], scale=std_deviation * shadow_radius[0], size=num_points);
            Y = np.random.normal(loc=shadow_coordinate[1], scale=std_deviation * shadow_radius[1], size=num_points);
            np.random.normal()

            rad_angle = np.deg2rad(angle)
            X_rotated, Y_rotated = rotate_vector_around_center((X, Y), rad_angle, shadow_coordinate)

            for j in range(num_points):
                x = X_rotated[j]
                y = Y_rotated[j]
                points.append((int(x), int(y)))

            points_inside_ellipses.append(points)
            ellipses.append(new_ellipse)
            i += 1

    return points_inside_ellipses, ellipses

def rotate_and_scale(image, annot):
    scale_factor = np.random.randint(3, 5)
    angle =  np.random.randint(0, 360)
    annot_class = np.max(annot) + 1

    scaled_image = cv2.resize(image, (int(image.shape[0]/scale_factor), int(image.shape[1]/scale_factor)))
    rotated_image = rotate_image(scaled_image, angle)

    scaled_annot =- cv2.resize(annot, (int(annot.shape[0]/scale_factor), int(annot.shape[1]/scale_factor)))
    rotated_annot = rotate_image(scaled_annot, angle)
    rotated_annot[rotated_annot != 0] = annot_class

    return rotated_image, rotated_annot

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def calculate_histogram(image):
    histogram = np.zeros(256, dtype=int)
    for pixel in image.ravel():
        if pixel != 0:
            histogram[pixel] += 1
    return histogram

def calculate_cdf(histogram):
    cdf = np.cumsum(histogram)
    return cdf

def histogram_equalization(image):
    histogram = calculate_histogram(image)
    cdf = calculate_cdf(histogram)

    num_pixels = image.size
    image_equalized = (cdf[image] * 255.0 / num_pixels).astype(np.uint8)
    
    return image_equalized

def match_histogram(source, target):
    source_histogram = calculate_histogram(source)
    source_cdf = calculate_cdf(source_histogram)
    
    target_histogram = calculate_histogram(target)
    target_cdf = calculate_cdf(target_histogram)

    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping[i] = np.argmin(np.abs(source_cdf[i] - target_cdf))
    
    matched_image = mapping[source]
    
    return matched_image

def split_RGB(image):
    return image[:,:,0], image[:,:,1], image[:,:,2] 

def match_rgb_histogram(source, target):
    source_R, source_G, source_B = split_RGB(source)
    target_R, target_G, target_B = split_RGB(target)

    matched_R = match_histogram(source_R, target_R)
    matched_G = match_histogram(source_G, target_G)
    matched_B = match_histogram(source_B, target_B)

    return np.dstack((matched_R, matched_G, matched_B))

def get_masked_image(image, mask):
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    return masked_image

def generate_weeds(
    groups: list,
    species:list,
    images: list, 
    masks: list,
    annotations: list,
    corns: list,
    corn_masks = list
):
    corn_index = np.random.randint(0, len(corns) - 1)
    corn_image = cv2.imread(corns[corn_index])
    corn_mask = cv2.imread(corn_masks[corn_index], cv2.IMREAD_GRAYSCALE)
    corn_annot = corn_mask.copy()
    corn_annot[corn_annot != 0] = 1

    all_weeds = np.zeros_like(corn_image)
    all_annots = np.zeros((corn_image.shape[0], corn_image.shape[1]), dtype=np.uint8)
    masked_corns = get_masked_image(corn_image, corn_mask)

    for points in groups:
        specie = species[np.random.randint(0, len(species) - 1)] if len(species) > 1 else species[0]
        _images = [f for f in images if specie in f]
        _masks = [f for f in masks if specie in f]
        _annots = [f for f in annotations if specie in f]

        for point in points:
            image_index = np.random.randint(0, len(_images) - 1)
            weed = cv2.imread(_images[image_index])
            mask = cv2.imread(_masks[image_index], cv2.IMREAD_GRAYSCALE)
            annot = cv2.imread(_annots[image_index], cv2.IMREAD_GRAYSCALE)
            weed, annot = rotate_and_scale(get_masked_image(weed, mask), annot)

            x_offset = (point[0], point[0] + weed.shape[1])
            y_offset = (point[1], point[1] + weed.shape[0])

            if x_offset[0] >= 0 and x_offset[1] <= all_weeds.shape[1] and y_offset[0] >= 0 and y_offset[1] <= all_weeds.shape[0]:
                all_weeds[y_offset[0]: y_offset[1], x_offset[0]: x_offset[1], :][~np.all(weed == [0,0,0], axis=-1)] = weed[~np.all(weed == [0,0,0], axis=-1)]
                all_annots[y_offset[0]: y_offset[1], x_offset[0]: x_offset[1]][annot != 0] = annot[annot != 0]

    final_image = corn_image.copy()
    final_image[~np.all(all_weeds == [0,0,0], axis=-1)] = all_weeds[~np.all(all_weeds == [0,0,0], axis=-1)] # add weeds
    final_image[~np.all(masked_corns == [0,0,0], axis=-1)] = masked_corns[~np.all(masked_corns == [0,0,0], axis=-1)] # add corns over the weeds 
    final_image = match_rgb_histogram(final_image, corn_image)
    all_annots[corn_annot != 0] = corn_annot[corn_annot != 0]

    return final_image, all_annots