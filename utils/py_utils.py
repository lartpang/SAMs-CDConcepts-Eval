from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label


def load_image(path: Path):
    image = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    assert image is not None, f"{path} is not a valid image."
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(path: Path, return_uint8: bool = False):
    mask = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
    assert mask is not None, f"{mask} is not a valid mask."

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary = mask.astype(bool)

    if return_uint8:
        return binary, mask
    return binary


def find_connect_area(mask):
    """Find connected components in a binary mask and return their bounding boxes."""
    label_mask, num_components = label(mask, structure=np.ones((3, 3)))
    bbox = []
    for connected_label in range(1, num_components + 1):
        component_corrds = np.where(label_mask == connected_label)
        min_x = np.min(component_corrds[1])
        min_y = np.min(component_corrds[0])
        max_x = np.max(component_corrds[1])
        max_y = np.max(component_corrds[0])
        bbox.append(np.array([min_x, min_y, max_x, max_y]))
    return bbox


def generate_whole_bbox(mask):
    """Find the bounding box of a binary mask.

    Args:
        mask (numpy.bool): binary mask

    Returns:
        Tuple[numpy.ndarray]: whole bbox
    """
    nonzero_coords = np.nonzero(mask)
    if nonzero_coords[0].size == 0 or nonzero_coords[1].size == 0:
        bbox = np.array([0, 0, 0, 0])
    else:
        min_x = np.min(nonzero_coords[1])
        min_y = np.min(nonzero_coords[0])
        max_x = np.max(nonzero_coords[1])
        max_y = np.max(nonzero_coords[0])
        bbox = np.array([min_x, min_y, max_x, max_y])
    return bbox


def perturb_bounding_boxes(bbox, img_height, img_width):
    """Perturbs the input bounding boxes by adding random errors to each boundary.

    Args:
        bbox (List[List[int]]): A list of bounding boxes, where each bounding box is a list containing [min_x, min_y, max_x, max_y].
        img_height (int): The height of the image.
        img_width (int): The width of the image.

    Returns:
        List[numpy.ndarray]: A list of perturbed bounding boxes, where each bounding box is a numpy array containing [new_min_x, new_min_y, new_max_x, new_max_y].
    """
    robust_bbox = []
    for box in bbox:
        min_x, min_y, max_x, max_y = box
        width = max_x - min_x
        height = max_y - min_y

        # 计算短边并定义误差范围
        short_edge = min(width, height)
        error_range = 0.1 * short_edge

        # 为每个边界生成不同的随机误差
        min_x_error = np.random.uniform(-error_range, error_range)
        max_x_error = np.random.uniform(-error_range, error_range)
        min_y_error = np.random.uniform(-error_range, error_range)
        max_y_error = np.random.uniform(-error_range, error_range)

        # 分别扰动每个边界
        new_min_x = min_x + min_x_error
        new_max_x = max_x + max_x_error
        new_min_y = min_y + min_y_error
        new_max_y = max_y + max_y_error

        # 保证边界框不超出图片大小
        new_min_x = np.clip(new_min_x, 0, img_width - 1)
        new_max_x = np.clip(new_max_x, 0, img_width - 1)
        new_min_y = np.clip(new_min_y, 0, img_height - 1)
        new_max_y = np.clip(new_max_y, 0, img_height - 1)

        robust_bbox.append(np.array([new_min_x, new_min_y, new_max_x, new_max_y]))
    return robust_bbox


def random_erose_or_dilate(mask, max_iterations=5):
    """对二值化mask进行随机腐蚀或膨胀测试，只返回一张处理后的mask。

    参数:
        mask (numpy.ndarray): 二值化的输入mask (bool类型)
        max_iterations (int): 最大腐蚀/膨胀的迭代次数

    Returns:
        numpy.ndarray: 随机选择腐蚀或膨胀后处理的mask
    """
    # 确保输入是bool类型的二值化mask
    mask = mask.astype(bool)
    # 随机选择腐蚀的迭代次数并进行腐蚀
    iterations = np.random.randint(1, max_iterations + 1)
    # 随机选择进行腐蚀还是膨胀
    if np.random.rand() > 0.5:
        mask = binary_erosion(mask, iterations=iterations)
    else:
        mask = binary_dilation(mask, iterations=iterations)
    return mask
