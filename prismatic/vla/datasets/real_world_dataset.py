import numpy as np
import torch
import os
import glob
import h5py
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import fnmatch
import subprocess
import pickle
import re
from datetime import datetime
import cv2
import logging
from PIL import Image
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision
import random


logger = logging.getLogger(__name__)
# Example
language_tasks = [
    'Put the screwdriver in the cabinet and close the cabinet',
]

class HDF5Dataset(torch.utils.data.Dataset):
    
    def __init__(self, episode_ids, 
                dataset_dir, 
                camera_names, 
                norm_stats, 
                window_size = 16,
                min_window_size = 16,
                max_window_size = 16,
                image_transform = None,
                other_config=()) -> None:
        
        super(HDF5Dataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.other_config = other_config
        self.chunk_size = window_size
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.resize_img = torchvision.transforms.Resize((224, 224))
        self.image_transform_lam = torchvision.transforms.ToTensor()
        self.image_transform = image_transform
        self.image_dict, self.qpos, self.action, self.tasks_embedding = self.load_all_episodes(dataset_dir)
        self.color_aug = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)


    def __len__(self):
        return len(self.action)

    def load_all_episodes(self, dataset_paths):
        image_dict = dict()
        image_hdf5_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = []
        qpos = []
        actions = []
        instructions = []
        for dataset_path in dataset_paths:
            print(f"processing {dataset_path}")

            with h5py.File(dataset_path, 'r') as root:
                compressed = root.attrs.get('compress', False)
                original_action_shape = root['/action'].shape
                self.episode_len = original_action_shape[0]

                qpos.append(np.array(root['/observations/qpos']))
                actions.append(np.array(root['/action']))
                
                file_name = dataset_path.split('/')[-1]

                # TODO: We store file names as task instructions, please adjust accordingly
                task_instruction = file_name.split('+')[0].replace('_', ' ')
                instructions.append(task_instruction)
                
                for cam_name in self.camera_names:
                    image_hdf5_dict[cam_name] = root[f'/observations/images/{cam_name}']
                for cam_name in image_dict.keys():
                    image_one_cam = []
                    for i_img in range(image_hdf5_dict[cam_name].shape[0]):
                        if compressed:
                            raw_image = cv2.imdecode(image_hdf5_dict[cam_name][i_img], 1)  # [480, 640, 3]
                        else:
                            raw_image = image_hdf5_dict[cam_name][i_img]
                        flipped_image = torch.flip(torch.from_numpy(raw_image), dims=(-1,))
                        resized_image = F.interpolate(flipped_image.permute(2, 0, 1).unsqueeze(0).float(), size=(224, 224), mode='bilinear', align_corners=False)
                        image_one_cam.append(resized_image[0])
                    image_dict[cam_name].append(torch.stack(image_one_cam, dim=0))
        for cam_name in self.camera_names:
            image_dict[cam_name] = torch.stack(image_dict[cam_name], dim=0)

        qpos = torch.from_numpy(np.stack(qpos, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()

        return image_dict, qpos, actions, instructions


    def __getitem__(self, clip_index):

        extra_frame_num = random.randint(0, 1)
        window_size = self.window_size + extra_frame_num
        
        image_index = np.random.choice(self.episode_len - window_size)
        actions_chunking = torch.zeros((self.chunk_size, self.action.shape[-1]))
        is_not_padding = torch.zeros((self.chunk_size,))
        
        actions_chunking[:min(self.episode_len-image_index, self.chunk_size)] = self.action[clip_index, image_index:image_index+min(self.episode_len-image_index, self.chunk_size)]
        qpos_chunking = self.qpos[clip_index][image_index]

        # cam_name = "0"
        cam_name = "camera_high"
        image_chunking = self.image_dict[cam_name][clip_index][image_index:image_index + window_size]
        image_vla = Image.fromarray(np.transpose(image_chunking[extra_frame_num].cpu().numpy().astype(np.uint8), (1, 2, 0)))
        image_vla = self.color_aug(image_vla)
        goal_image = Image.fromarray(np.transpose(image_chunking[-1].cpu().numpy().astype(np.uint8), (1, 2, 0)))
        pixel_values = self.image_transform(image_vla)
        
        initial_pixel_values = self.image_transform_lam(self.resize_img(image_vla))
        target_pixel_values = self.image_transform_lam(self.resize_img(goal_image))
        
        initial_pixel_values_hist, target_pixel_values_hist = None, None
        if extra_frame_num > 0:
            hist_frame_prev = Image.fromarray(np.transpose(image_chunking[0].cpu().numpy().astype(np.uint8), (1, 2, 0)))
            hist_frame_goal = Image.fromarray(np.transpose(image_chunking[self.min_window_size].cpu().numpy().astype(np.uint8), (1, 2, 0)))
            initial_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_prev))
            target_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_goal))
        
        is_not_padding[:min(self.episode_len-image_index, self.chunk_size)] = 1
        
        # normalize actions and change dtype to float
        qpos_tensor = qpos_chunking.float()
        action_tensor = actions_chunking.float()
        action_tensor = (action_tensor - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_tensor = (qpos_tensor - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        task_embed = self.tasks_embedding[clip_index]
        
        dataset_name = 'agilex'
        
        return dict(pixel_values=pixel_values, initial_pixel_values=initial_pixel_values, target_pixel_values=target_pixel_values, 
                    initial_pixel_values_hist=initial_pixel_values_hist, target_pixel_values_hist=target_pixel_values_hist,
                    dataset_name=dataset_name, actions=action_tensor, lang=task_embed, proprio=qpos_tensor)


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]

        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(instance["initial_pixel_values_hist"])
                target_pixel_values_hist.append(instance["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))     



        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None


        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            actions=actions,
            proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


def load_data_univla(dataset_paths, camera_names, batch_size_train, action_tokenizer, processor, window_size,     
        min_window_size, max_window_size, image_transform, other_info=()):

    num_episodes = len(dataset_paths)
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_paths, other_info)

    train_dataset = HDF5Dataset(train_indices, dataset_paths, camera_names, norm_stats,
        window_size = window_size,
        min_window_size = min_window_size,
        max_window_size = max_window_size,
        image_transform = image_transform,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=8, prefetch_factor=2, collate_fn=collator,)


    return train_dataloader, norm_stats


def find_all_hdf5(dataset_dir, skip_mirrored_data=True):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def get_norm_stats(dataset_paths, other_config=()):
    all_qpos_data = []
    all_action_data = []
    for dataset_path in dataset_paths:
        #dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            if 'qvel' in other_config:
                qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    # all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    
    # Min-max norm action datra
    action_max = all_action_data.max(dim=0, keepdim=True)[0][0]   # torch.Size([58200, 7])
    action_min = all_action_data.min(dim=0, keepdim=True)[0][0]

    # print(action_max.shape, action_min)


    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "action_max":action_max, "action_min":action_min}

    print(stats)

    return stats

def get_key_info(path):
    if '.pkl' not in path:
        path = os.path.join(path, f'key_info.pkl')
    with open(path, 'rb') as f:
        key_info = pickle.load(f)
    return key_info

def get_init_states(path_first_episode):
    if os.path.exists(path_first_episode):
        with h5py.File(path_first_episode, 'r') as root:
            qpos = root['/observations/qpos'][0]
            action = root['/action'][0]
    else:
        # dir is info dir
        key_info_path = os.path.join(dir, f'key_info.pkl')
        with open(key_info_path, 'rb') as f:
            key_info = pickle.load(f)
            qpos = key_info['init_info']['init_joint']
            action = key_info['init_info']['init_action']
    return qpos, action


