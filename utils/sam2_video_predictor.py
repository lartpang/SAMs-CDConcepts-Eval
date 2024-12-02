import os
from collections import OrderedDict

import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import AsyncVideoFrameLoader, _load_img_as_tensor
from tqdm import tqdm


def load_image_with_prompts(
    image_path,
    prompt_image_paths,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the images from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(image_path, str) and os.path.isfile(image_path):
        jpg_path = image_path
    else:
        raise NotImplementedError("Only JPEG frames are supported at this moment")

    assert isinstance(prompt_image_paths, (list, tuple)) and len(prompt_image_paths) > 0
    img_paths = prompt_image_paths
    # target frame is placed at the final index
    assert jpg_path.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png"))
    img_paths.append(jpg_path)

    num_frames = len(img_paths)

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(img_paths, image_size, offload_video_to_cpu, img_mean, img_std)
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


class ICLSAM2VideoPredictor(SAM2VideoPredictor):
    @torch.inference_mode()
    def init_group_prompts(
        self,
        prompt_image_paths,
        offload_video_to_cpu=False,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
    ) -> tuple:
        assert isinstance(prompt_image_paths, (list, tuple)) and len(prompt_image_paths) > 0
        num_prompt_images = len(prompt_image_paths)

        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        prompt_images = torch.zeros(num_prompt_images, 3, self.image_size, self.image_size, dtype=torch.float32)
        for n, img_path in enumerate(tqdm(prompt_image_paths, desc="frame loading (JPEG)")):
            prompt_images[n], video_height, video_width = _load_img_as_tensor(img_path, self.image_size)

        if not offload_video_to_cpu:
            prompt_images = prompt_images.cuda()
            img_mean = img_mean.cuda()
            img_std = img_std.cuda()

        # normalize by mean and std
        prompt_images -= img_mean
        prompt_images /= img_std
        return prompt_images

    def load_single_image(
        self,
        image_path,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
    ):
        if isinstance(image_path, str) and os.path.isfile(image_path):
            jpg_path = image_path
        else:
            raise NotImplementedError("Only JPEG frames are supported at this moment")

        assert jpg_path.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png"))

        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        if async_loading_frames:
            lazy_images = AsyncVideoFrameLoader([image_path], self.image_size, offload_video_to_cpu, img_mean, img_std)
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.float32)
        images[0], video_height, video_width = _load_img_as_tensor(image_path, self.image_size)

        if not offload_video_to_cpu:
            images = images.cuda()
            img_mean = img_mean.cuda()
            img_std = img_std.cuda()

        # normalize by mean and std
        images -= img_mean
        images /= img_std
        return images, video_height, video_width

    @torch.inference_mode()
    def init_state(
        self,
        image_path,
        group_prompts,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize a inference state."""
        images, video_height, video_width = self.load_single_image(
            image_path=image_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        images = torch.cat([group_prompts, images], dim=0)

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state
