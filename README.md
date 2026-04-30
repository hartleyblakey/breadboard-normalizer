# Task 2: Breadboard Normalizer

**Usage**:

```python
import cv2
import numpy as np
from PIL import Image

from breadboard_normalizer.normalizer import Normalizer, draw_corners, resize_width

normalizer = Normalizer(padding=[0.0, 0.0], output_resolution=[1024, 340])

image = Image.open("example_training_images/4b6949fe-0ae765d4-69E2A987-8D4F-4075-A989-01B6BA302390.jpeg")


# normalizer assumes RGB
image = np.asarray(image)

# flip it to test the orientation detection
image = np.flipud(image)

norm, source_corners = normalizer.normalize_image(image)

if norm is None:
    print("Failed to normalize image")
else:
    # CV2 assumes BGR
    norm = np.flip(norm, axis=-1)
    image = np.flip(image, axis=-1)

    image = draw_corners(image, source_corners)

    norm = draw_corners(norm, normalizer.destination_corners)

    image = resize_width(image, normalizer.target_size[0])

    image = np.vstack([image, norm])

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```