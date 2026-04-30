from pathlib import Path

import tensorflow as tf
from numpy.typing import ArrayLike
import numpy as np
from docaligner import DocAligner
import cv2
from PIL import Image
import os


def draw_corners(image, corners, radius=None, thickness=None):
    # red, green, blue, magenta
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
    width_diff = abs(np.linalg.norm(corners[0] - corners[2]) - np.linalg.norm(corners[1] - corners[3]))
    return length_diff + width_diff;

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


class Normalizer:

    _corner_rough_model = None

    _corner_flip_model = None

    target_size: np.ndarray = np.array([1024, 340])

    corner_flip_class_names = ['flipped', 'correct', 'obstructed', 'missed']

    pad: np.ndarray = np.array([0.00, 0.00])

    destination_corners: np.ndarray = np.array([
        [target_size[0] * pad[0], target_size[1] * pad[1]],
        [target_size[0] * (1.0 - pad[0]), target_size[1] * pad[1]],
        [target_size[0] * (1.0 - pad[0]), target_size[1] * (1.0 - pad[1])],
        [target_size[0] * pad[0], target_size[1] * (1.0 - pad[1])],
    ], dtype=np.float32)

    corner_size: int = 32
    model_pad = np.array([0.00, 0.00])
    corner_fill: float = 1.0

    def __init__(self, padding=None, output_resolution=None):
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

    def find_raw_corners(self, image):
        source_corners = self._corner_rough_model(image)

        if len(source_corners) != 4:
            return None
        
        return reorder_corners(source_corners)


    def normalize_image_raw(self, image):
        source_corners = self._corner_rough_model(image)

        if len(source_corners) != 4:
            return None
        
        source_corners = reorder_corners(source_corners)

        transform = cv2.getPerspectiveTransform(source_corners, self.destination_corners)

        return cv2.warpPerspective(image, transform, dsize=self.target_size)
    
    def normalize_image(self, image):
        norm = self.normalize_image_raw(image)
        label = self.breadboard_orientation_cv(norm)
        if label == 'flipped':
            norm = np.flipud(norm)
        return norm

    _image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')



    def _predict_corners_cv(self, norm_bgr):




        pass

    def _show_ml_annotated_image(self, file, window_name):
        image = Image.open(file)
        image = np.asarray(image)
        print(image.shape)

        normalized_image = self.normalize_image_raw(image)

        if normalized_image is None:
            print(f"Failed to find corners at {file}")
            return

        norm_bgr = np.flip(normalized_image, axis=-1)

        annotated = draw_corners(norm_bgr, self.destination_corners)

        corner_crops = self.crop_corners(norm_bgr)

        corner_flip_predictions = self._corner_flip_model.predict(corner_crops, verbose=0)

        for i in range(4):
            index = np.argmax(corner_flip_predictions[i])
            label = self.corner_flip_class_names[index]
            corner_crops[i] = cv2.putText(corner_crops[i], label, (4, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


        corner_stack = resize_width(np.hstack(corner_crops), self.target_size[0])

        total_pred = np.sum(corner_flip_predictions[:, :2], axis=0)
        index = np.argmax(total_pred)
        label = self.corner_flip_class_names[index]

        annotated = cv2.putText(annotated, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)

        annotated = np.vstack([annotated, corner_stack])

        cv2.imshow(window_name, annotated)


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

    def _show_annotated_image(self, file, window_name):
        image = Image.open(file)
        image = np.asarray(image)
        print(image.shape)

        normalized_image = self.normalize_image_raw(image)

        if normalized_image is None:
            print(f"Failed to find corners at {file}")
            return

        norm_bgr = np.flip(normalized_image, axis=-1)

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

        norm_bgr_scaled = norm_float.astype(np.uint8)

        annotated_red = annotated_red.astype(np.uint8)

        annotated_blue = annotated_blue.astype(np.uint8)

        # print(annotated.shape)

        # annotated = cv2.resize(norm_bgr, (8, 256), interpolation=cv2.INTER_AREA)

        annotated_red = cv2.resize(annotated_red, [norm_bgr.shape[1], norm_bgr.shape[0]], interpolation=cv2.INTER_NEAREST)

        annotated_blue = cv2.resize(annotated_blue, [norm_bgr.shape[1], norm_bgr.shape[0]], interpolation=cv2.INTER_NEAREST)

        image_resized = resize_width(image, norm_bgr.shape[1])
        image_resized = np.flip(image_resized, axis=-1) # convert to BGR
        corners = self.find_raw_corners(image_resized)
        if corners is not None:
            image_resized = draw_corners(image_resized, corners)

        norm_bgr_flipped = norm_bgr

        if label == 'flipped':
            norm_bgr_flipped = np.flipud(norm_bgr_flipped)


        annotated = np.vstack([image_resized, norm_bgr, annotated_red + annotated_blue, norm_bgr_flipped])

        annotated = cv2.putText(annotated, label, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        annotated = cv2.putText(annotated, label, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        cv2.imshow(window_name, annotated)


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
                    self._show_annotated_image(entry.path, window_name)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
        
        cv2.destroyAllWindows()
        

    









