from pathlib import Path

import tensorflow as tf
from numpy.typing import ArrayLike
import numpy as np
from docaligner import DocAligner
import cv2
from PIL import Image, ImageOps
import os
from fastquadtree import QuadTree
from pycpd import RigidRegistration, AffineRegistration
from typing import Literal, Optional


import numpy as np

try:
    import open3d as o3d
    USE_OPEN3D = True
except: 
    USE_OPEN3D = False


import random
from skimage import transform

# from https://github.com/ClayFlannigan/icp
# edited to allow arrays of different length
from .vendor.icp import icp

def draw_corners(image, corners, radius=None, thickness=None, colors=None):
    # red, green, blue, magenta
    if colors is None:
        colors=[[0,0,255], [0, 255, 0], [255, 0, 0], [255, 0, 255]]
    corners = np.array(corners)

    annotated = np.copy(image)

    if radius == None:
        radius = np.min(image.shape[:2]) // 40

    if thickness == None:
        thickness = radius * 2

    for i, corner in enumerate(corners):
        annotated = cv2.circle(annotated, corner.astype(int), radius=radius, color=colors[i % 4], thickness=thickness)
        annotated = cv2.putText(annotated, f"{i}", (int(corner[0]) - radius // 2, int(corner[1]) + radius // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    return annotated


def is_landscape(corners):
    even_dist = np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[2] - corners[3]) 
    odd_dist = np.linalg.norm(corners[1] - corners[2]) + np.linalg.norm(corners[3] - corners[0]) 

    return even_dist > odd_dist

    # center = corners.mean(axis=0)

    # dif = (corners - center)
    # a_dif = np.abs(dif).mean(axis=0)

    # return a_dif[0] > a_dif[1]

def reorder_corners(corners):

    center = corners.mean(axis=0)

    relative = corners - center

    angles = -np.atan2(relative[:, 1], relative[:, 0])

    indices = angles.argsort()

    corners = corners[indices]

    portrait_indices = np.array([1, 2, 3, 0])

    if not is_landscape(corners):
        corners = corners[portrait_indices]

    return corners


def skew_metric(corners):
    length_diff = abs(np.linalg.norm(corners[0] - corners[1]) - np.linalg.norm(corners[2] - corners[3]))
    width_diff = abs(np.linalg.norm(corners[0] - corners[3]) - np.linalg.norm(corners[1] - corners[2]))
    return length_diff + width_diff;

def aspect_metric(corners):
    target_aspect_ratio = 1024/340
    length = np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[2] - corners[3])
    width =  np.linalg.norm(corners[0] - corners[3]) + np.linalg.norm(corners[1] - corners[2])
    return abs(length / width - target_aspect_ratio)

def crop_square(image, l, size):
    h, w = image.shape[:2]
    l[0] = min(l[0], w - size - 1)
    l[1] = min(l[1], h - size - 1)
    return image[l[1]:l[1]+size, l[0]:l[0]+size]

def resize_width(image: np.ndarray, target_width: int):
    h, w = image.shape[:2]
    factor = float(target_width) / float(w)
    return cv2.resize(image, (target_width, int(h * factor)), interpolation=cv2.INTER_NEAREST)

def resize_height(image: np.ndarray, target_height: int):
    h, w = image.shape[:2]
    factor = float(target_height) / float(h)
    return cv2.resize(image, (int(w * factor), target_height), interpolation=cv2.INTER_AREA)

def get_source_corners_from_label(filename):
    source_corners = np.loadtxt(filename, dtype=np.float32)[1:].reshape(4 ,2)
    return reorder_corners(source_corners)

def image_label_pairs(image_dir, label_dir):
    for index, file in enumerate(os.listdir(os.fsencode(image_dir))):
        filename, ext = os.path.splitext(os.fsdecode(file))
        if ext.lower().endswith(Normalizer._image_extensions):
            image_pil = Image.open(image_dir + filename + ext)
            image_pil = ImageOps.exif_transpose(image_pil)
            image = np.array(image_pil)
            label = get_source_corners_from_label(label_dir + filename + '.txt')
            yield filename, image, label


def normalize_points(points):
    normalized = np.copy(points)

    translation = -np.mean(normalized, axis=0)

    normalized += translation

    max_dist = np.max(np.abs(normalized)) 
    scale_val = 1.0 / max_dist if max_dist != 0 else 1.0
    scale = np.array([scale_val, scale_val])

    normalized *= scale

    m = np.eye(3)
    m[0, 0] = scale[0]
    m[1, 1] = scale[1]

    m[:2, 2] = translation * scale

    return normalized, m


class PinGrid:
    _size: np.ndarray
    _pad: np.ndarray

    _base_points: np.ndarray
    points: np.ndarray
    labels = None

    _quadtree: QuadTree

    def __init__(self, size: np.ndarray, padding: np.ndarray=np.array([0, 0])):
        self._size = size
        self._pad = padding
        self._base_points, self.labels = PinGrid.base_pin_holes()
        
        print("Creating pingrid with", len(self._base_points), "points and", len(self.labels), "labels")
        
        low = self._pad * self._size
        high = self._size + low
        padded_base_points = self._base_points / (1.0 + padding * 2) + padding
        self.points = np.array(padded_base_points * self._size, dtype=np.float32)
        self._quadtree = QuadTree((low[0], low[1], high[0], high[1]), capacity=16)
        self._quadtree.insert_many_np(self.points)
    
    
    def transform_points_3x3(points, matrix):
        points_transformed = points.reshape(-1, 1, 2)
        points_transformed = cv2.perspectiveTransform(points_transformed, matrix)
        return points_transformed.reshape(-1, 2)
    
    def nearest_neighbors(self, points):
        target_correspondences = []
        for point in points:
            _, c = self._quadtree.nearest_neighbor_np((point[0], point[1]))
            target_correspondences.append(c)
        return np.array(target_correspondences, dtype=np.float32)


    def fit_icp(self, source: np.ndarray):
        if USE_OPEN3D:
            pass
        else:
            pass
        h, distances, iterations = icp(source, self.points, max_iterations=50)
        return h
    


    def rigid_cpd(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = RigidRegistration(X=t_n, Y=s_n, w=0.2, sigma2=0.05)
        transformed, ((scale, rotation, translation)) = reg.register()
        h = np.eye(3)
        h[:2, 2] = translation
        h[:2, :2] = scale * rotation

        t_m_inv = np.linalg.inv(t_m)

        # h maps from s_m to t_m
        # make it go from s_m to world
        h = t_m_inv @ h

        # make it go from world to world
        h = h @ s_m

        transformed = PinGrid.transform_points_3x3(transformed, t_m_inv)

        return (transformed, h)

    def rigid_cpd_rev(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = RigidRegistration(X=s_n, Y=t_n, w=0.2, sigma2=0.05)
        transformed, ((scale, rotation, translation)) = reg.register()
        h = np.eye(3)
        h[:2, 2] = translation
        h[:2, :2] = scale * rotation

        # s_m to t_m
        h = np.linalg.inv(h)

        s_m_inv = np.linalg.inv(s_m)

        # h maps from s_m to t_m
        # make it go from t_m to world
        h = s_m_inv @ h

        # make it go from world to world
        h = h @ t_m

        transformed = PinGrid.transform_points_3x3(source, h)

        return (transformed, h)

    def affine_cpd(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = AffineRegistration(X=t_n, Y=s_n, w=0.1, sigma2=0.05)
        transformed, ((affine, translation)) = reg.register()
        
        affine_3 = np.eye(3)
        affine_3[:2, :2] = affine.T # apparently you need to transpose here for handedness or something, according to chatGPT

        h = np.eye(3)
        h[:2, 2] = translation

        h = h @ affine_3

        t_m_inv = np.linalg.inv(t_m)

        # h maps from s_m to t_m
        # make it go from s_m to world
        h = t_m_inv @ h

        # make it go from world to world
        h = h @ s_m

        transformed = PinGrid.transform_points_3x3(transformed, t_m_inv)

        return (transformed, h)

    def affine_cpd_rev(source, target):
        # s_n, s_m = normalize_points(source)
        t_n, t_m = normalize_points(target)
        s_m = t_m
        s_n = PinGrid.transform_points_3x3(source, t_m)

        reg = AffineRegistration(X=s_n, Y=t_n, w=0.05, sigma2=0.05)
        _, ((affine, translation)) = reg.register()
        
        affine_3 = np.eye(3)
        affine_3[:2, :2] = affine.T # apparently you need to transpose here for handedness or something, according to chatGPT

        h = np.eye(3)
        h[:2, 2] = translation

        h = h @ affine_3

        h = np.linalg.inv(h)

        t_m_inv = np.linalg.inv(t_m)

        # h maps from normalized source to normalized target
        # make it go from normalized source to input coordinates
        h = t_m_inv @ h

        # make it go input coordinates to input coordinates
        h = h @ s_m

        transformed = PinGrid.transform_points_3x3(source, h)

        return (transformed, h)

    def fit_brute_force(self, source: np.ndarray):
        """
        Guess random initial transformations, use nearest neighbors to guess at correspondences,
        and run it through cv2.findHomography() 30 times and keep the transform with the most inliners

        findHomography() is the only method here that handles perspective, but it needs 1-1 correspondences
        between points in the source and target arrays. Nearest neighbors alone are a terrible way to do
        this because of grid aliasing, and this method almost always fits one side well and lets the other 
        side explode into perspective distortions.
        """
        
        spacing = np.mean(self._size / np.array([65.1, 21.25]))
        best_matches = 0
        best_transform = np.eye(3)

        for i in range(30):
            if i == 0:
                guess_transform= np.eye(3)
            else:
                translation = np.array([
                    random.randrange(-2, 3), random.randrange(-2, 3)
                    ]) * spacing
                translation = np.array([
                    random.random(), random.random()
                    ]) * spacing
                rotation = (random.random() - 0.5) * 0.01
                scale = np.array([random.random(), random.random()]) * 0.05 + 0.975

                guess_transform = transform.AffineTransform(
                    scale=scale,
                    rotation=rotation,
                    translation=translation
                ).params
            # pick 3 points and guess at correspondence



            source_transformed = PinGrid.transform_points_3x3(source, guess_transform)

            target_correspondences = self.nearest_neighbors(source_transformed)

            h, mask = cv2.findHomography(target_correspondences, source_transformed, method=cv2.RANSAC, ransacReprojThreshold=spacing/175.0, confidence=0.995)
            if h is None or mask is None:
                continue

            h = np.linalg.inv(h)

            inliners = np.count_nonzero(mask)
            if inliners > best_matches:
                best_matches = inliners
                best_transform = h @ guess_transform
        return best_transform
    
    def fit_brute_force_icp(self, source: np.ndarray):
        """
        Try random *Horizontal* transformations, then apply ICP and keep the best. 

        More reliable than fit_brute_force() since its more constrained and the error metric
        is closer aligned to the problem. Intended to fix grid aliasing caused by other fitting
        methods.
        """
        
        spacing = np.mean(self._size / np.array([65.1, 21.25]))
        best_score = 999.0
        best_transform = np.eye(3)

        for i in range(30):
            if i == 0:
                guess_transform= np.eye(3)
            else:
                translation = np.array([
                    random.randrange(-8, 12), 0
                    ]) * spacing / 4.0
                rotation = (random.random() - 0.5) * 0.01
                rotation = 0.0
                scale = np.array([random.random(), 1.0]) * 0.05 + 0.975

                guess_transform = transform.AffineTransform(
                    scale=scale,
                    rotation=rotation,
                    translation=translation
                ).params
            # pick 3 points and guess at correspondence

            source_transformed = PinGrid.transform_points_3x3(source, guess_transform)

            h = self.fit_icp(source_transformed)

            source_refined = PinGrid.transform_points_3x3(source_transformed, h)

            rmse, duplicates, inliners = self.evaluate_fit(source_refined)
            score = (rmse / (1.0 - duplicates))
            if score < best_score:
                best_score = score
                best_transform = h @ guess_transform
        return best_transform

    def eval_rmse(src, tgt):
        assert src.shape == tgt.shape
        # no idea how or why this works https://stackoverflow.com/questions/21926020/how-to-calculate-rmse-using-ipython-numpy
        return np.linalg.norm(src - tgt) / np.sqrt(len(src))

    def evaluate_fit(self, pts: np.ndarray):
        neighbors = self.nearest_neighbors(pts)

        distances = np.linalg.norm(pts - neighbors, axis=1)

        inliner_mask = distances < (self._size[0] / 65.0) * 0.25

        inliners = pts[inliner_mask]
        inliner_neighbors = neighbors[inliner_mask]

        rmse = PinGrid.eval_rmse(inliners, inliner_neighbors)

        # rescale so its resolution agnostic and roughly in units of pin holes

        rmse_grid = rmse / self._size[0]
        rmse_grid *= 65

        # more than one detected pinhole maps to a target pinhole
        # Attempt to catch off-by-one scale problems that might optimize RMSE on noisy inputs
        duplicates = len(neighbors) - len(np.unique(neighbors, axis=0))

        # 1 is perfect, 0 is completely degenerate
        duplicate_score = 1.0 - duplicates/len(neighbors) + (1.0 / len(neighbors))
        
        inliner_ratio = len(inliners) / len(pts)

        return rmse_grid, inliner_ratio, duplicates/len(neighbors)

    def closest_grid_cell(self, pos: np.ndarray):
        ''' 
        Returns a (point, label) pair, or None if there was an internal error 
        '''
        res = self._quadtree.nearest_neighbor_np(tuple(pos.flatten()))
        if res is None:
            return None
        i, point = res
        return point, self.labels[i]

    def label_to_string(label):
        grid_name, x, y = label
        if grid_name == 'base_top':
            letters = ['a', 'b', 'c', 'd', 'e']
            return f"Center grid at {x+1}{letters[y]}"
        if grid_name == 'base_bot':
            letters = ['f', 'g', 'h', 'i', 'j']
            return f"Center grid at {x+1}{letters[len(letters)-y-1]}"
        if grid_name == 'rail_top':
            polarity = ['-', '+']
            return f"Top rail at {x+1}{polarity[1-y]}"
        if grid_name == 'rail_bot':
            polarity = ['-', '+']
            return f"Bottom rail at {x+1}{polarity[y]}"
        return "Unknown grid name"

    def grid_points(x, y):
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.stack([X, Y], axis=-1).reshape(-1, 2)

    def base_pin_holes():
        """
        A rough manually tuned grid. 
        Made by visually aligning the points to a stacked image of the training data with pinhole detections highlighted.

        Returns a points, labels pair, where each label is a (grid_name, x, y) pair.

        There are 4 grids for the 4 connected grids on the breadboard (The center grid halves are not connected to each other)
        """
        labels = []
    
        bb_size = np.array([65.1, 21.25])
        center_pos = np.array([1.525, 5.15])
        center_base = PinGrid.grid_points(np.arange(63), np.arange(5)).astype(float)
        center_labels = []
        for i, (x, y) in enumerate(center_base):
            center_labels.append(('base_top', int(x), int(y)))
        center_base += center_pos

        rail_pos = np.array([3.525, 1.35])
        rail_x = np.zeros((50, 1))
        for i in range(59):
            rail_x[i - i//6] = i
        rail_base = PinGrid.grid_points(rail_x, np.arange(2)) + rail_pos
        rail_labels = []
        for x, _ in enumerate(rail_x):
            for y in range(2):
                rail_labels.append(('rail_top', x, y))
        
        pts = np.concatenate((rail_base, center_base), axis=0)
        labels = rail_labels + center_labels

        pts_copy = np.copy(pts)
        pts_copy[:, 1] = bb_size[1] - pts_copy[:, 1]

        labels_copy = labels.copy()
        for i in range(len(labels_copy)):
            name, x, y = labels_copy[i]
            if name == 'base_top':
                labels_copy[i] = ('base_bot', x, y)
            if name == 'rail_top':
                labels_copy[i] = ('rail_bot', x, y)

        pts = np.concatenate((pts, pts_copy), axis=0)

        labels = labels + labels_copy

        return pts / bb_size, labels


class Normalizer:
    """
    A class to handle extracting the breadboard from an image.
    Makes a base assumption that the breadboard exists and is the focus of the photo.
    """

    _corner_rough_model = None

    _corner_flip_model = None

    target_size: np.ndarray = np.array([1024, 340])

    # corner_flip_class_names = ['flipped', 'correct', 'obstructed', 'missed']
    corner_flip_class_names = ['corner', 'invalid']

    # not sure how else to return so many values in an ergonomic way
    # (inliner_rmse, inliner_ratio, duplicate_ratio)
    last_score = None

    pad: np.ndarray = np.array([0.00, 0.00])

    pingrid: PinGrid

    destination_corners: np.ndarray = np.array([
        [target_size[0] * pad[0], target_size[1] * pad[1]],
        [target_size[0] * (1.0 - pad[0]), target_size[1] * pad[1]],
        [target_size[0] * (1.0 - pad[0]), target_size[1] * (1.0 - pad[1])],
        [target_size[0] * pad[0], target_size[1] * (1.0 - pad[1])],
    ], dtype=np.float32)

    corner_size: int = 32
    model_pad = np.array([0.00, 0.00])
    corner_fill: float = 1.0


    RegistrationMethod = Optional[Literal["affine_cpd", "rigid_cpd", "icp"]]
    

    def __init__(self, padding=None, output_resolution=None, raw_pingrid: PinGrid = None):

        if raw_pingrid is not None:
            assert padding is None or padding == raw_pingrid._pad
            assert output_resolution is None or output_resolution == raw_pingrid._size
            self.pingrid = raw_pingrid
            padding = raw_pingrid._pad
            output_resolution = raw_pingrid._size

        if padding is not None:
            if isinstance(padding, float):
                self.pad = np.array([padding, padding])
            else:
                self.pad = np.array(padding)
        if output_resolution is not None:
            if isinstance(output_resolution, float):
                self.target_size = (self.target_size.astype(float) * output_resolution).astype(int)
            else:
                self.target_size = np.array(output_resolution).astype(int)

        if raw_pingrid is None:
            self.pingrid = PinGrid(self.target_size, self.pad)
        
        self.destination_corners: np.ndarray = np.array([
            [self.target_size[0] * self.pad[0], self.target_size[1] * self.pad[1]],
            [self.target_size[0] * (1.0 - self.pad[0]), self.target_size[1] * self.pad[1]],
            [self.target_size[0] * (1.0 - self.pad[0]), self.target_size[1] * (1.0 - self.pad[1])],
            [self.target_size[0] * self.pad[0], self.target_size[1] * (1.0 - self.pad[1])],
        ], dtype=np.float32)
                
        self._corner_rough_model = DocAligner()
        self.destination_corners = reorder_corners(self.destination_corners)
        src_dir = Path(__file__).parent.parent
        model_path = src_dir / "weights" / "corner_orientation.keras"
        self._corner_flip_model = tf.keras.models.load_model(model_path)
        return
    
    def crop_corners(self, image):
        """Returns an an array containing the 4 square corners of the image, cropped according to corner_size"""
        h, w = (image.shape[0], image.shape[1])

        o = np.array([w, h]) * self.pad - self.corner_size * (1.0 - self.corner_fill)
        o = np.array(o, dtype=int)
        oi = np.array([w, h]) * (1.0 - self.pad) - self.corner_size * self.corner_fill
        oi = np.array(oi, dtype=int)
        return np.array([
            crop_square(image, [o[0], oi[1]], self.corner_size),
            crop_square(image, [oi[0], oi[1]], self.corner_size),
            crop_square(image, [oi[0], o[1]], self.corner_size),
            crop_square(image, [o[0], o[1]], self.corner_size),
    ])

    def find_corners(self, image):
        """
        Finds the corners in an image. Returned values are in pixels, and the corners are in the following order,
        with respect to the shape in the image and not the orientation of the breadboard

        3------2\n
        0------1

        Returns None if the model failed to find all 4 corners
        """
        source_corners = self._corner_rough_model(image)

        if len(source_corners) != 4:
            return None
        
        return reorder_corners(source_corners)

    def warp_image(self, image, corners):
        """
        Warps the image so the provided corners are mapped to Normalizer.destination_corners. 
        """

        transform = cv2.getPerspectiveTransform(corners, self.destination_corners)

        return cv2.warpPerspective(image, transform, dsize=self.target_size), transform

    def find_refinement_transform(self, norm_rough, registration: RegistrationMethod = 'affine_cpd'):
        """
        Returns the refinement transform in output space
        """
        keypoints = Normalizer.find_circles(norm_rough)

        source = []
        for keypoint in keypoints:
            source.append(keypoint.pt)
        source = np.array(source)

        if registration == "affine_cpd":
            transformed, h = PinGrid.affine_cpd_rev(source, self.pingrid.points)
        elif registration == "rigid_cpd":
            transformed, h = PinGrid.rigid_cpd_rev(source, self.pingrid.points)
        elif registration == "icp":
            h = self.pingrid.fit_icp(source)
            transformed = PinGrid.transform_points_3x3(source, h)

        rmse, inliner_ratio, dup_ratio  = self.pingrid.evaluate_fit(transformed)

        return h, (rmse, inliner_ratio, dup_ratio)

    def normalize_image(self, image, registration: RegistrationMethod = 'affine_cpd'):
        """
        Returns an (image, corners) pair where:
        - image is the normalize image of size self.target_size, with the corners of the breadboard at
          self.destination_corners
        - the positive rail is on top
        - corners is the pixel-space position of the corners in the original image
          - so corners[0] and corners[1] are always the bottom of the breadboard, with the negative rail on the bottom
        """

        source_corners = self.find_corners(image)
        if source_corners is None:
            return (None, None, None)

        norm_rough, h = self.warp_image(image, source_corners)

        output_refinement, h_score = self.find_refinement_transform(norm_rough, registration=registration)

        last_score_full = h_score

        inliner_rmse, inliner_ratio, duplicate_ratio = h_score

        refined_h = output_refinement @ h

        score = 1.0

        # score is an error metric
        if inliner_rmse > 0.1 or inliner_ratio < 0.85 or duplicate_ratio > 0.075:
            # refined_h = h
            score = 0.75
            print(f"Warning: refinement transform had an inliner rmse of {inliner_rmse}, inliner ratio of {inliner_ratio}, duplicate ratio of {duplicate_ratio}")
        if inliner_rmse > 0.15 or inliner_ratio < 0.8 or duplicate_ratio > 0.15:
            refined_h = h
            score = 0.0
            print(f"Refinement transform failed.")
            print(f"Warning: refinement transform had an inliner rmse of {inliner_rmse}, inliner ratio of {inliner_ratio}, duplicate ratio of {duplicate_ratio}")
        
        norm = cv2.warpPerspective(image, refined_h, dsize=self.target_size)

        # this is kind of sketchy
        source_corners = PinGrid.transform_points_3x3(self.destination_corners, np.linalg.inv(refined_h))

        
        label = self.breadboard_orientation_cv(norm)

        if label == 'flipped':
            norm = np.rot90(norm, k=2)
            source_corners = np.roll(source_corners, shift=2, axis=0)

        return norm, source_corners, score

    _image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


    def _show_ml_annotated_image(self, image, window_name):

        source_corners = self.find_corners(image)

        image_bgr = np.flip(image, axis=-1)

        if source_corners is None:
            return image_bgr

        normalized_image = self.warp_image(image, source_corners)


        norm_bgr = np.flip(normalized_image, axis=-1)

        

        corner_crops = self.crop_corners(norm_bgr)

        corner_flip_predictions = self._corner_flip_model.predict(corner_crops, verbose=0)

        # for i in range(4):
        #     index = np.argmax(corner_flip_predictions[i])
        #     label = self.corner_flip_class_names[index]
        #     corner_crops[i] = cv2.putText(corner_crops[i], label, (4, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        corner_crops_resized = np.zeros((4, self.corner_size * 4, self.corner_size * 4, 3))

        label_indices = corner_flip_predictions[:, 0] > 0.5
        label_colors = [(0, 0, 255), (0, 255, 0)]
        corner_colors = []
        for i in label_indices:
            corner_colors.append(label_colors[int(i)])

        print(corner_flip_predictions)
        print(label_indices)


        # for i in range(4):
        #     index = corner_flip_predictions[:][0] > 0.5
        #     label = self.corner_flip_class_names[index]
        #     corner_crops_resized[i] = cv2.resize(corner_crops[i], (self.corner_size * 4, self.corner_size * 4), cv2.INTER_NEAREST) / 256
        #     corner_crops_resized[i] = cv2.putText(corner_crops_resized[i], label, (4, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # corner_stack = resize_width(np.hstack(corner_crops_resized), self.target_size[0] * 4)

        annotated = draw_corners(image_bgr, source_corners, colors=corner_colors)


        annotated = resize_width(annotated, 1920)

        skew = 1.0 - (skew_metric(source_corners) / image.shape[0]) * 5
        aspect = 1.0 - aspect_metric(source_corners)
        metric = (skew * 0.5 + 0.5) * aspect

        if label_indices.astype(int).sum() < 3:
            metric *= 0.75

        # total_pred = np.sum(corner_flip_predictions[:, :2], axis=0)
        # index = np.argmax(total_pred)
        # label = self.corner_flip_class_names[index]
        box_color = (0, 0, 100)
        if metric > 0.5:
            box_color = (0, 100, 0)
        cv2.rectangle(annotated, (10, 10), (1200, 260), box_color, -1)
        cv2.putText(annotated, f"Skew: {skew:.2f}", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(annotated, f"Aspect: {aspect:.2f}", (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(annotated, f"Validation Metric: {metric:.2f}", (16, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        # annotated = np.vstack([annotated, corner_stack])

        cv2.imshow(window_name, annotated)

        return annotated

    def find_circles(image, show_images=  False):
        # tuned for np.array([1024, 340])
        base_size = np.array([1024, 340])

        h, w = image.shape[:2]

        scale = w / base_size[0]
        image_resized = resize_width(image, base_size[0])


        params = cv2.SimpleBlobDetector_Params()
        
        # Thresholds for binarization
        params.minThreshold = 50
        params.maxThreshold = 150
        
        params.filterByArea = True
        params.minArea = 10
        
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 0.95
        
        params.filterByConvexity = True
        params.minConvexity = 0.95
        
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # params.filterByColor = True
        # params.blobColor = 0
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        image_float = np.copy(image_resized).astype(np.float32)
        blur = cv2.blur(image_float, (64, 64))

        image_float = image_float / blur
        image = np.clip(image_float * 155, 0, 255).astype(np.uint8)


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # blur image
        # blur = cv2.GaussianBlur(gray, (5,5), 0)

        # # do otsu threshold on gray image
        # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,5)
        # thresh = cv2.GaussianBlur(thresh, (5,5), 0)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Apply to image
        clahe_img = clahe.apply(gray)

        # edges = cv2.Canny(clahe_img, 50, 200, None, 3)
        # edges = np.copy(clahe_img)
        # thresh = edges > 50
        # inv_thresh = edges<=50
        # edges[inv_thresh] = 0
        # edges[thresh] = 255

        # skeleton = skimage.morphology.skeletonize(edges)

        # edges[skeleton] = 255
        # edges[~skeleton] = 0

        # lines = cv2.HoughLinesP(edges, 3, np.pi / 80, 20, None, 50, 10)
        
        # line_img = np.zeros((image.shape[0] * 3, image.shape[1] * 3, 3))
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         l = lines[i][0] * 3
        #         cv2.line(line_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)

        if show_images:
            cv2.imshow("Annotated image 2", clahe_img)
            cv2.waitKey(0)
            # cv2.imshow("Annotated image 2", edges)
            # cv2.waitKey(0)
            # cv2.imshow("Annotated image 2", line_img)
            # cv2.waitKey(0)
            # Detect blobs
        keypoints = detector.detect(clahe_img)

        for kp in keypoints:
            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
            kp.size *= scale

        return keypoints

    def breadboard_orientation_cv(self, image):
        norm_bgr = np.flip(image, axis=-1)

        avg = cv2.blur(norm_bgr, (64, 64))

        norm_float = norm_bgr.astype(np.float32)
        norm_float /= avg
        norm_float *= 255

        norm_float = np.clip(norm_float, 0, 255)

        length = np.linalg.norm(norm_float, axis=2, keepdims=True)
        length = np.nan_to_num(length) + 0.001
        normalized = norm_float / length
        zeros = np.zeros_like(norm_float[:, :1, 0])

        red = (np.dot(normalized, np.array([0, 0, 1])))
        red -= np.mean(red)
        red = cv2.blur(red, (3, 3))
        red = np.median(red, axis=1)[:, np.newaxis]
        red /= np.max(red)
        annotated_red = red * 255.0
        annotated_red = np.clip(annotated_red, 0, 255)
        annotated_red = np.stack((zeros, zeros, annotated_red), axis=-1)

        blue = (np.dot(normalized, np.array([1, 0, 0])))
        blue -= np.mean(blue)
        blue = cv2.blur(blue, (3, 3))
        blue = np.median(blue, axis=1)[:, np.newaxis]
        blue /= np.max(blue)
        annotated_blue = blue * 255.0
        annotated_blue = np.clip(annotated_blue, 0, 255)
        annotated_blue = np.stack((annotated_blue, zeros, zeros), axis=-1)

        red = red.flatten()
        blue =  blue.flatten()

        crop_size = int(len(red) * 0.33)

        red_top = red[:crop_size]
        red_bot = red[-crop_size:]

        blue_top = blue[:crop_size]
        blue_bot = blue[-crop_size:]

        red_top_peak = np.argmax(red_top)
        blue_top_peak = np.argmax(blue_top)

        red_bot_peak = np.argmax(red_bot)
        blue_bot_peak = np.argmax(blue_bot)

        top_vote = np.sign(blue_top_peak - red_top_peak) * red_top[red_top_peak] * blue_top[blue_top_peak]
        bot_vote = np.sign(blue_bot_peak - red_bot_peak) * red_bot[red_bot_peak] * blue_bot[blue_bot_peak]

        vote = np.sign(top_vote + bot_vote)
        confidence = np.max((np.abs(top_vote), np.abs(bot_vote)))

        label = "unknown"

        if confidence >= 0.01:
            if vote == -1:
                label = "flipped"
            elif vote == 1:
                label = "correct"
            elif vote == 0:
                label = "disputed"
            else:
                label = "np.sign was not -1, 0 or 1"
        
        return label

    def __filter_tails(v: np.ndarray, l: int = 15):
        b = v[0]
        t = v[-1]

        m = np.mean(v)
        for i in range(0, l):
            if v[i] > 1.2 * b or v[i] < 0.8 * t:
                break
            v[i] = m
        
        for i in reversed(range(len(v) - l, len(v))):
            if v[i] > 1.2 * t or v[i] < 0.8 * t:
                break
            v[i] = m
            
            
        return v

    def _show_annotated_image(self, image, window_name):

        source_corners = self.find_corners(image)

        if source_corners is None:
            return None

        normalized_image = self.warp_image(image, source_corners)

        if normalized_image is None:
            return None

        norm_bgr = np.flip(normalized_image, axis=-1)

        avg = cv2.blur(norm_bgr, (128, 128))

    

        norm_float = norm_bgr.astype(np.float32)
        norm_float /= avg
        norm_float *= 255 / 2

        norm_float = np.clip(norm_float, 0, 255)


        length = np.linalg.norm(norm_float, axis=2, keepdims=True)
        length = np.nan_to_num(length) + 0.01
        normalized = norm_float / length


        # hsv = cv2.cvtColor(norm_float, cv2.COLOR_BGR2HSV)
        # mask = hsv[:, :, 1] <= 0.1
        

        red = normalized[:, :, 2]
        # red[mask] = 0
        red -= np.mean(red)
        red = cv2.blur(red, (3, 3))
        red_pre_median = np.clip(np.copy(red) * 255, 0, 255).astype(np.uint8)
        red = np.median(red, axis=1)[:, np.newaxis]
        red /= np.max(red)
        annotated_red = red * 255.0
        annotated_red = np.clip(annotated_red, 0, 255)

        blue = normalized[:, :, 0]
        # blue[mask] = 0
        blue -= np.mean(blue)
        blue = cv2.blur(blue, (3, 3))
        blue_pre_median = np.clip(np.copy(blue) * 255, 0, 255).astype(np.uint8)
        blue = np.median(blue, axis=1)[:, np.newaxis]
        blue /= np.max(blue)
        annotated_blue = blue * 255.0
        annotated_blue = np.clip(annotated_blue, 0, 255)
        

        # red = norm_float[:, :, 2] / total
        # red[mask] = 0
        # red -= np.mean(red)
        # red = cv2.blur(red, (3, 3))
        # red_pre_median = red
        # red = np.median(red, axis=1)[:, np.newaxis]
        # red /= np.max(red)
        # annotated_red = red * 255.0
        # annotated_red = np.clip(annotated_red, 0, 255)    

        # blue = norm_float[:, :, 0] / total
        # blue[mask] = 0
        # blue -= np.mean(blue)
        # blue = cv2.blur(blue, (3, 3))
        # blue = np.median(blue, axis=1)[:, np.newaxis]
        # blue /= np.max(blue)
        # annotated_blue = blue * 255.0
        # annotated_blue = np.clip(annotated_blue, 0, 255)

        red = red.flatten()
        blue =  blue.flatten()

        crop_size = int(len(red) * 0.33)

        red_top = Normalizer.__filter_tails(red[:crop_size])
        red_bot = Normalizer.__filter_tails(red[-crop_size:])

        blue_top = Normalizer.__filter_tails(blue[:crop_size])
        blue_bot = Normalizer.__filter_tails(blue[-crop_size:])

        red_top_peak = np.argmax(red_top)
        blue_top_peak = np.argmax(blue_top)

        red_bot_peak = np.argmax(red_bot)
        blue_bot_peak = np.argmax(blue_bot)

        top_vote = np.sign(blue_top_peak - red_top_peak) * red_top[red_top_peak] * blue_top[blue_top_peak]
        bot_vote = np.sign(blue_bot_peak - red_bot_peak) * red_bot[red_bot_peak] * blue_bot[blue_bot_peak]

        vote = np.sign(top_vote + bot_vote)
        confidence = np.max((np.abs(top_vote), np.abs(bot_vote)))

        label = "unknown"

        if confidence >= 0.1:
            if vote == -1:
                label = "flipped"
            elif vote == 1:
                label = "correct"
            elif vote == 0:
                label = "disputed"
            else:
                label = "np.sign was not -1, 0 or 1"

        norm_bgr_scaled = norm_float.astype(np.uint8)

        norm_bgr_flipped = norm_bgr

        if label == 'flipped':
            norm_bgr_flipped = np.flipud(norm_bgr_flipped)


        annotated_red = annotated_red.astype(np.uint8)

        annotated_blue = annotated_blue.astype(np.uint8)

        # print(annotated.shape)

        # annotated = cv2.resize(norm_bgr, (8, 256), interpolation=cv2.INTER_AREA)

        annotated_red = cv2.resize(annotated_red, [norm_bgr.shape[1], norm_bgr.shape[0]], interpolation=cv2.INTER_NEAREST)

        annotated_blue = cv2.resize(annotated_blue, [norm_bgr.shape[1], norm_bgr.shape[0]], interpolation=cv2.INTER_NEAREST)

        normalized[:, :, 1] = 0
        normalized = np.flip(normalized, axis=-1)
        normalized *= 16

        pre_median_vis =  np.stack([blue_pre_median, np.zeros_like(annotated_blue), red_pre_median], axis=-1)
        pre_median_vis = np.clip(pre_median_vis * 8, 0, 255)

        annotated = np.vstack([norm_float.astype(np.uint8), pre_median_vis, np.stack((annotated_blue, np.zeros_like(annotated_blue), annotated_red), axis=-1)])




        image_resized = resize_height(image, annotated.shape[0])

        factor = image_resized.shape[0] / image.shape[0]

        image_resized = np.flip(image_resized, axis=-1) # convert to BGR
        if source_corners is not None:
            image_resized = draw_corners(image_resized, source_corners * factor)




        annotated = np.hstack([image_resized, annotated])

        annotated = cv2.putText(annotated, label, (36, 122), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 16)
        annotated = cv2.putText(annotated, label, (38, 124), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)

        annotated = np.vstack([annotated, resize_width(norm_bgr_flipped, annotated.shape[1])])

        cv2.imshow(window_name, annotated)
        return annotated


    def visualize_model(self, path):

        window_name = "Annotated image"

        if not os.path.exists(path):
            print(f"Failed to find path at {path}")
            return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if os.path.isfile(path):
            self._show_annotated_image(path, window_name)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(self._image_extensions):
                    image = Image.open(entry.path)
                    image = np.asarray(image)
                    if self._show_ml_annotated_image(image, window_name) is None:
                        print(f"Failed to find corners at {entry.path}")
                        
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
        
        cv2.destroyAllWindows()


        

    









