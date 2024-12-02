from copy import deepcopy

import cv2
import numpy as np
import torch


class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 255
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        assert pred_mask.ndim == 2, pred_mask.shape
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 255

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    return intersection / union


def get_points_nd(clicks_lists):
    total_clicks = []
    num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
    num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
    num_max_points = max(num_pos_clicks + num_neg_clicks)
    num_max_points = max(1, num_max_points)

    for clicks_list in clicks_lists:
        pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

        neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
    return total_clicks


def get_ideal_points(clicker, reverse=True):
    clicks_list = clicker.get_clicks()
    points_nd = get_points_nd([clicks_list])
    point_length = len(points_nd[0]) // 2
    point_coords = []
    point_labels = []
    for i, point in enumerate(points_nd[0]):
        if point[0] == -1:
            continue
        if i < point_length:
            point_labels.append(1)
        else:
            point_labels.append(0)
        if reverse:
            point_coords.append([point[1], point[0]])  # for SAM
    return np.array(point_coords), np.array(point_labels)


def get_perturbed_points(clicker, img_height, img_width, reverse=True):
    clicks_list = clicker.get_clicks()
    points_nd = get_points_nd([clicks_list])
    point_length = len(points_nd[0]) // 2
    point_coords = []
    point_labels = []
    for i, point in enumerate(points_nd[0]):
        if point[0] == -1:
            continue
        if i < point_length:
            point_labels.append(1)
        else:
            point_labels.append(0)

        # 处理点坐标顺序
        if reverse:
            x, y = point[1], point[0]  # For SAM

        # 生成随机扰动范围 [-10, 10]
        x_error = np.random.uniform(-10, 10)
        y_error = np.random.uniform(-10, 10)

        # 加入扰动并进行边界检查
        new_x = np.clip(x + x_error, 0, img_width - 1)
        new_y = np.clip(y + y_error, 0, img_height - 1)

        point_coords.append([new_x, new_y])
    return np.array(point_coords), np.array(point_labels)


@torch.no_grad()
def prompt_sams_with_point(
    image, gt_mask, predictor, max_iou_thr, min_clicks=1, max_clicks=20, prompt_mask=None, perturbation=False
):
    """Performs interactive point clicking to generate a mask.

    Args:
        image (numpy.ndarray): _description_
        gt_mask (numpy.ndarray): 0 is background, 255 is foreground, and -1 will be ignored.
        predictor (nn.Module): _description_
        max_iou_thr (float): _description_
        min_clicks (int, optional): _description_. Defaults to 1.
        max_clicks (int, optional): _description_. Defaults to 20.
        prompt_mask (numpy.ndarray, optional): _description_. Defaults to None.
        perturbation (bool, optional): _description_. Defaults to False.

    Returns:
        numpy.ndarray: _description_
    """
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask) if prompt_mask is None else prompt_mask

    predictor.set_image(image)
    for click_idx in range(max_clicks):
        clicker.make_next_click(pred_mask)

        if perturbation:
            point_coords, point_labels = get_perturbed_points(clicker, gt_mask.shape[0], gt_mask.shape[1])
        else:
            point_coords, point_labels = get_ideal_points(clicker)

        pred_probs = predictor.predict(point_coords, point_labels, multimask_output=True, return_logits=True)[0]

        pred_masks = []
        ious = []
        for prob in pred_probs:
            mask = prob > 0.0
            pred_masks.append(mask)
            ious.append(get_iou(gt_mask, mask))
        tgt_idx = np.argmax(np.array(ious))
        pred_mask = pred_masks[tgt_idx]
        if ious[tgt_idx] >= max_iou_thr and click_idx + 1 >= min_clicks:
            break
    return pred_mask


@torch.no_grad()
def prompt_sam2_with_point(
    gt_mask, predictor, max_iou_thr, inference_state, frame_idx, min_clicks=1, max_clicks=20, perturbation=False
):
    """Performs interactive point clicking to generate a mask.

    Args:
        gt_mask (numpy.ndarray): 0 is background, 255 is foreground, and -1 will be ignored.
        predictor (_type_): _description_
        max_iou_thr (_type_): _description_
        inference_state (_type_): _description_
        frame_idx (_type_): _description_
        min_clicks (int, optional): _description_. Defaults to 1.
        max_clicks (int, optional): _description_. Defaults to 20.
        perturbation (bool, optional): _description_. Defaults to False.
    """
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)

    for click_idx in range(max_clicks):
        clicker.make_next_click(pred_mask)

        if perturbation:
            point_coords, point_labels = get_perturbed_points(clicker, gt_mask.shape[0], gt_mask.shape[1])
        else:
            point_coords, point_labels = get_ideal_points(clicker)

        pred_probs = predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=frame_idx, obj_id=0, points=point_coords, labels=point_labels
        )[-1]

        pred_masks = []
        ious = []
        for pred_prob in pred_probs:
            mask = pred_prob.squeeze().cpu().numpy() > 0.0
            pred_masks.append(mask)
            ious.append(get_iou(gt_mask, mask))
        tgt_idx = np.argmax(np.array(ious))
        pred_mask = pred_masks[tgt_idx]
        if ious[tgt_idx] >= max_iou_thr and click_idx + 1 >= min_clicks:
            break
