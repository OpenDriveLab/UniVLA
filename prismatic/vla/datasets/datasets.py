"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# From 3Hz to 5Hz control frequency
datasets_with_lower_frequency = [
    "fractal20220817_data",
    "toto",
    "berkeley_autolab_ur5",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "dobbe",
]

# From 15Hz to 30 Hz control frequency
datasets_with_higher_frequency = [
    "utaustin_mutex",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "toto",
    "viola",
    "droid",
]


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    aug_transform: Optional[Callable] = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        obs_frames = rlds_batch["observation"]["image_primary"]
        img = Image.fromarray(obs_frames[0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # print(labels)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # Create augmented view with brightness/contrast jitter and optional frame shift
        if self.aug_transform is not None:
            shift_idx = 0
            if len(obs_frames) > 1:
                shift = random.choice([-1, 0, 1])
                shift_idx = max(0, min(len(obs_frames) - 1, shift))
            img_aug = Image.fromarray(obs_frames[shift_idx])
            img_aug = self.aug_transform(img_aug)
            pixel_values_aug = self.image_transform(img_aug) + 0.01 * torch.randn_like(pixel_values)
        else:
            pixel_values_aug = pixel_values.clone()

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            pixel_values_aug=pixel_values_aug,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
        )


@dataclass
class RLDSBatchTransformLIBERO_withHis:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 5

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        randomized_overlap = random.randint(0, 1)
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][self.window_size - 1])

        input_img = Image.fromarray(rlds_batch["observation"]["image_primary"][randomized_overlap])
        pixel_values = self.image_transform(input_img)

        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(input_img)
            target_pixel_values = self.image_transform_lam(
                Image.fromarray(rlds_batch["observation"]["image_primary"][self.window_size - 1 + randomized_overlap])
            )

            video = (
                torch.stack([initial_pixel_values, target_pixel_values], dim=0)
                .unsqueeze(0)
                .to(self.action_tokenizer.device)
            )
            latent_action_idx = self.action_tokenizer.vq_encode(video)["indices"].squeeze()

            if randomized_overlap > 0:
                initial_pixel_values = self.image_transform_lam(img)
                target_pixel_values = self.image_transform_lam(img_k)
                video = (
                    torch.stack([initial_pixel_values, target_pixel_values], dim=0)
                    .unsqueeze(0)
                    .to(self.action_tokenizer.device)
                )
                hist_action_idx = self.action_tokenizer.vq_encode(video)["indices"].squeeze()

        action_vocab = [f"<ACT_{i.item()}>" for i in latent_action_idx]  # [ACT_1, ACT_2, ... ACT_K]
        # print(action_vocab)
        action_tokens = ""
        for _, action in enumerate(action_vocab):
            action_tokens += action

        input_prompt = f"What action should the robot take to {lang}?"
        if randomized_overlap > 0:
            action_vocab = [f"<ACT_{i.item()}>" for i in hist_action_idx]  # [ACT_1, ACT_2, ... ACT_K]

            hist_action_tokens = ""
            for _, action in enumerate(action_vocab):
                hist_action_tokens += action

            input_prompt = f"What action should the robot take to {lang}? History action " + hist_action_tokens

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": input_prompt},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=rlds_batch["action"][randomized_overlap : self.window_size + randomized_overlap],
            latent_action_idx=latent_action_idx,
            dataset_name=dataset_name,
        )


@dataclass
class RLDSBatchTransformLIBERO:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        obs_frames = rlds_batch["observation"]["image_primary"]
        img = Image.fromarray(obs_frames[0])
        img_k = Image.fromarray(obs_frames[-1])
        pixel_values = self.image_transform(img)

        # Create augmented view with brightness/contrast jitter and optional frame shift
        if self.aug_transform is not None:
            shift_idx = 0
            if len(obs_frames) > 1:
                shift = random.choice([-1, 0, 1])
                shift_idx = max(0, min(len(obs_frames) - 1, shift))
            img_aug = Image.fromarray(obs_frames[shift_idx])
            img_aug = self.aug_transform(img_aug)
            pixel_values_aug = self.image_transform(img_aug)
            pixel_values_aug = pixel_values_aug + 0.01 * torch.randn_like(pixel_values_aug)
        else:
            pixel_values_aug = pixel_values.clone()

        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            video = (
                torch.stack([initial_pixel_values, target_pixel_values], dim=0)
                .unsqueeze(0)
                .to(self.action_tokenizer.device)
            )
            latent_action_idx = self.action_tokenizer.vq_encode(video)["indices"].squeeze()

        action_vocab = [f"<ACT_{i.item()}>" for i in latent_action_idx]  # [ACT_1, ACT_2, ... ACT_K]

        action_tokens = ""
        for _, action in enumerate(action_vocab):
            action_tokens += action
        # print(action_tokens)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=rlds_batch["action"],
            latent_action_idx=latent_action_idx,
            dataset_name=dataset_name,
        )


@dataclass
class RLDSBatchTransformLatentAction:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    aug_transform: Optional[Callable] = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        obs_frames = rlds_batch["observation"]["image_primary"]
        img = Image.fromarray(obs_frames[0])
        img_k = Image.fromarray(obs_frames[-1])
        pixel_values = self.image_transform(img)

        if self.aug_transform is not None:
            shift_idx = 0
            if len(obs_frames) > 1:
                shift = random.choice([-1, 0, 1])
                shift_idx = max(0, min(len(obs_frames) - 1, shift))
            img_aug = Image.fromarray(obs_frames[shift_idx])
            img_aug = self.aug_transform(img_aug)
            pixel_values_aug = self.image_transform(img_aug) + 0.01 * torch.randn_like(pixel_values)
        else:
            pixel_values_aug = pixel_values.clone()

        with torch.no_grad():
            initial_pixel_values = self.image_transform_lam(img)
            target_pixel_values = self.image_transform_lam(img_k)
            video = (
                torch.stack([initial_pixel_values, target_pixel_values], dim=0)
                .unsqueeze(0)
                .to(self.action_tokenizer.device)
            )
            latent_action_idx = self.action_tokenizer.vq_encode(video)["indices"].squeeze()

        action_vocab = [f"<ACT_{i.item()}>" for i in latent_action_idx]  # [ACT_1, ACT_2, ... ACT_K]

        action_tokens = ""
        for _, action in enumerate(action_vocab):
            action_tokens += action

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            pixel_values_aug=pixel_values_aug,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
        )


@dataclass
class RLDSBatchTransformVideo:
    image_transform: ImageTransform

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], np.array(rlds_batch["action"])

        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])  # .copy()
        initial_pixel_values = self.image_transform(img)

        # the frame interval is already tackled in RLDS dataloader
        target_frame_index = -1
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][target_frame_index])  # .copy()
        # print(sum(np.array(img_k) - np.array(img)))
        target_pixel_values = self.image_transform(img_k)

        return dict(
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            task_instruction=lang,
            action=action,
            dataset_name=dataset_name,
        )


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        window_size: int = 10,
        train: bool = True,
        image_aug: bool = False,
        training_phase: str = "lam",
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=8,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            training_phase=training_phase,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
