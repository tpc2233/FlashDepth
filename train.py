import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, sys
from hydra.core.hydra_config import HydraConfig
import argparse
import yaml
from tqdm import tqdm
import ipdb
import random
from itertools import cycle
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision.utils import save_image 

from utils.init_setup import dist_init, setup_model, save_checkpoint
# from utils.helpers import 
import logging
from utils import logging_config

from dataloaders.combined_dataset import CombinedDataset
from dataloaders.random_dataset import RandomDataset
from dataloaders.get_separate_dataloaders import get_all_dataloaders

import hydra
from omegaconf import DictConfig, OmegaConf


@torch.no_grad()
def validation(cfg, model, train_step, test_dataloader):
    model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
 
    savepath = os.path.join(cfg.config_dir, "val")
    os.makedirs(savepath, exist_ok=True)

    # Initialize nested dictionary to store losses per dataset
    val_losses = {}

    for test_idx, batch in enumerate(tqdm(test_dataloader, disable=(rank != 0))): 
        video, gt_depth, dataset_name = batch
        
        # Initialize dataset entry if not exists
        if dataset_name not in val_losses:
            val_losses[dataset_name] = {}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            gif_path = f'{savepath}/{os.path.basename(cfg.config_dir.rstrip("/"))}_{train_step}_idx-{test_idx * world_size + rank}.gif'
            
            loss, grid = model(
                (video, gt_depth), 
                use_mamba=cfg.model.use_mamba, 
                out_mp4=cfg.eval.out_mp4,
                gif_path=gif_path,
                resolution=cfg.eval.save_res,  # resolution (height) for saved gif / mp4
                print_time=False, #(rank == 0), 
            )

        # Accumulate losses per dataset
        for key, value in loss.items():
            if key not in val_losses[dataset_name]:
                val_losses[dataset_name][key] = []
            val_losses[dataset_name][key].append(value)

    # Average losses per dataset
    for dataset_name in val_losses:
        for key in val_losses[dataset_name]:
            val_losses[dataset_name][key] = sum(val_losses[dataset_name][key]) / len(val_losses[dataset_name][key])
    
    # Reduce losses across all processes
    for dataset_name in val_losses:
        for key in val_losses[dataset_name]:
            val_tensor = torch.tensor(val_losses[dataset_name][key], device=torch.cuda.current_device())
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            val_losses[dataset_name][key] = (val_tensor / world_size).item()
 
    # Calculate and log average across all datasets
    avg_losses = {}
    for key in next(iter(val_losses.values())).keys():  # Get all loss types from first dataset
        avg_losses[key] = sum(dataset_losses[key] for dataset_losses in val_losses.values()) / len(val_losses)
    
    if rank == 0 and cfg.training.wandb:
        # Log per-dataset losses
        for dataset_name, losses in val_losses.items():
            wandb.log({f"{dataset_name}/{k}": v for k, v in losses.items()}, step=train_step)
        # Log average losses
        wandb.log({f"avg/{k}": v for k, v in avg_losses.items()}, step=train_step)
        
        if grid is not None:
            wandb.log({"vis_val": wandb.Video(grid['stacked_frames'][::5], fps=10, format="gif")}, step=train_step)
    
    model.train()
    dist.barrier()


@torch.no_grad()
def inference(cfg, process_dict):

    model, _, _, train_step = setup_model(cfg, process_dict)
    model.eval()
    logging.info(f"Inference from step {train_step}")

    if cfg.eval.compile:
        model = torch.compile(model) 

    base_savepath = os.path.join(cfg.config_dir, cfg.eval.outfolder)
    os.makedirs(base_savepath, exist_ok=True)

    save_png_sequence = cfg.eval.get('save_png_sequence', False)

    eval_args = {
        'save_depth_npy': cfg.eval.save_depth_npy,
        'save_vis_map': cfg.eval.save_vis_map,
        'out_video': cfg.eval.out_video,
        'out_mp4': cfg.eval.out_mp4,
        'use_mamba': cfg.model.use_mamba,
        'resolution': cfg.eval.save_res,
        'print_time': True,
        'loss_type': cfg.training.loss_type,
        'use_all_frames': True,
        'use_metrics': False,
        'dummy_timing': cfg.eval.dummy_timing
    }
    
    if cfg.eval.dummy_timing:
        test_dataloader = [torch.randn(1, 105, 3, 1148, 2044)]*3
    elif cfg.eval.random_input is not None:
        dataset = RandomDataset(root_dir=cfg.eval.random_input, resolution=cfg.dataset.resolution, large_dir=cfg.eval.large_dir)
        test_dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False) 
    else:
        dataset = CombinedDataset(root_dir=cfg.dataset.data_root, enable_dataset_flags=cfg.eval.test_datasets, 
                                  split='test', resolution=cfg.eval.test_dataset_resolution)
        test_dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False) 
        
    for test_idx, batch in enumerate(tqdm(test_dataloader)):
        scene_name_str = f"item_{test_idx}"
        if isinstance(batch, list) or isinstance(batch, tuple):
            if len(batch) == 3: video, gt_depth, dataset_scene_name = batch; scene_name_str = dataset_scene_name[0].replace('/', '_')
            else: video, dataset_scene_name = batch; gt_depth = None; scene_name_str = dataset_scene_name[0].replace('/', '_')
            savepath = os.path.join(base_savepath, scene_name_str); batch = (video, gt_depth); logging.info(f'batch shape: {batch[0].shape}')
        elif isinstance(batch, dict):
            batch, scene_name_str = batch['batch'], batch['scene_name'][0]; savepath = os.path.join(base_savepath, scene_name_str.replace('/', '_')); logging.info(f'shape: {batch.shape}, {scene_name_str}')
        else: savepath = os.path.join(base_savepath, scene_name_str); logging.info(f'shape: {batch.shape}')
        os.makedirs(savepath, exist_ok=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, grid_dict = model(batch, gif_path=f'{savepath}/tmp.gif', **eval_args)

        if save_png_sequence and grid_dict is not None and isinstance(grid_dict, dict):
            
            # 'stacked_frames' is the correct key.
            # It contains the side-by-side visualization grid.
            key_for_visual_grid = 'stacked_frames'

            if key_for_visual_grid not in grid_dict:
                 logging.error(f"The key '{key_for_visual_grid}' was not found in the model output! Aborting.")
                 continue

            visual_grid_output = grid_dict[key_for_visual_grid]
            
            # Convert to a float tensor for processing
            if isinstance(visual_grid_output, torch.Tensor):
                side_by_side_tensor = visual_grid_output.float()
            else: # Assumes numpy array
                side_by_side_tensor = torch.from_numpy(visual_grid_output).float()

            # --- THE CROP LOGIC ---
            # The tensor contains two images side-by-side. We want only the right half (the depth map).
            _, _, _, total_width = side_by_side_tensor.shape
            single_image_width = total_width // 2

            logging.info(f"Cropping side-by-side grid of width {total_width} to get depth map of width {single_image_width}.")
            
            # Slice the tensor to get the right half
            depth_only_tensor = side_by_side_tensor[:, :, :, single_image_width:]
            # --- END OF CROP LOGIC ---

            frames_folder = os.path.join(savepath, "frames")
            os.makedirs(frames_folder, exist_ok=True)

            num_frames = depth_only_tensor.shape[0]
            logging.info(f"Saving {num_frames} cropped depth frames to {frames_folder}...")

            for i in range(num_frames):
                current_frame = depth_only_tensor[i] 
                frame_path = os.path.join(frames_folder, f"frame_{str(i).zfill(5)}.png")
                save_image(current_frame, frame_path, normalize=True)


def main(cfg, process_dict):

    # init dataloader
    dataloaders = get_all_dataloaders(cfg)
    train_iterators = [iter(dataloader) for dataloader in dataloaders]
    dataloader_cycle = cycle(range(len(dataloaders)))
    logging.info(f'Loaded training datasets, {len(dataloaders)} dataloaders initialized...')

    test_dataset = CombinedDataset(root_dir=cfg.dataset.data_root, enable_dataset_flags=cfg.dataset.val_datasets,
                             split='val', resolution=cfg.dataset.resolution)
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, 
                                 shuffle=False, drop_last=True, sampler=test_sampler)

    # init model
    model, optimizer, lr_scheduler, train_step = setup_model(cfg, process_dict)
    logging.info(f"Starting training at step {train_step}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # init logging (wandb)
    wandb_mode = "online" if (process_dict['rank'] == 0 and cfg.training.wandb) else "disabled"
    wandb_project = cfg.training.wandb_name
    wandb.init(
        mode=wandb_mode,
        project=wandb_project,
        name=os.path.basename(cfg.config_dir.rstrip("/")),
        config=dict(cfg)
    )
    wandb.watch(model, log='all', log_freq=100)

    if cfg.training.start_with_val:
        validation(cfg, model, train_step, test_dataloader)

    # start training loop
    total_iters = cfg.training.total_iters; epoch = 1
    pbar = tqdm(total=total_iters, initial=train_step, disable=(dist.get_rank() != 0))
    optimizer.zero_grad()
    for train_idx in range(total_iters-train_step):
        current_loader_idx = next(dataloader_cycle)
        try:
            batch = next(train_iterators[current_loader_idx])
        except StopIteration:
            epoch += 1
            dataloaders[current_loader_idx].sampler.set_epoch(train_step)
            train_iterators[current_loader_idx] = iter(dataloaders[current_loader_idx])
            batch = next(train_iterators[current_loader_idx])
        
        video, gt_depth, dataset_name = batch
        dataset_name = dataset_name[0] if isinstance(dataset_name, tuple) else dataset_name
        dataset_name = 'total' if isinstance(dataset_name, list) else dataset_name

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            vis_training = train_step if train_idx % cfg.training.vis_freq == 0 else 0
            vis_training = train_idx+1 if train_idx<10 else vis_training
            if cfg.model.use_mamba:    
                loss, grid = model.module.train_sequence((video, gt_depth), loss_type=cfg.training.loss_type, timestep=train_step,
                 vis_training=vis_training, savedir=os.path.join(cfg.config_dir, 'train_vis_debug'))
            else:
                loss, grid = model.module.train_single((video, gt_depth), loss_type=cfg.training.loss_type, 
                vis_training=vis_training, savedir=os.path.join(cfg.config_dir, 'train_vis_debug'))

            if vis_training and grid is not None:
                wandb.log({"vis_train_grid": wandb.Image(grid['grid_img'])}, step=train_step)
            loss = loss / cfg.training.gradient_accumulation

        loss.backward()

        if (train_idx+1) % cfg.training.gradient_accumulation == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_step += 1
            pbar.update(1)

        if (train_idx % 10000 == 0 or train_step in [0, cfg.training.lr.warmup_steps]) and dist.get_rank()==0:
            for i, param_group in enumerate(optimizer.param_groups):
                logging.info(f"Step {train_step}, Group {i + 1} LR: {param_group['lr']:.2e}")

        pbar.set_description(f"training - epoch: {epoch}, loss of {dataset_name}: {loss.detach().item()*cfg.training.gradient_accumulation:.3f}")
        if cfg.training.wandb and process_dict['rank'] == 0: 
            loss_details = {'loss': {}}
            loss_details['loss'][dataset_name] = loss.detach().item()*cfg.training.gradient_accumulation 
            wandb.log(loss_details, step=train_step)  

        if train_idx!=0 and train_idx % cfg.training.save_freq == 0:
            save_checkpoint(cfg, model, optimizer, lr_scheduler, train_step)
            dist.barrier()

        if train_idx!=0 and train_idx % cfg.training.val_freq == 0:
            validation(cfg, model, train_step, test_dataloader)

    wandb.finish()
    pbar.close()

@hydra.main(config_path=None, config_name="config", version_base="1.3")
def setup(cfg: DictConfig):
    
    process_dict = dist_init()
    logging_config.configure_logging()
    
    hydra_cfg = HydraConfig.get()
    cfg.config_dir = [path["path"] for path in hydra_cfg.runtime.config_sources if path["schema"] == "file"][0]
 
    if cfg.inference:
        inference(cfg, process_dict)
    else:
        main(cfg, process_dict)

    dist.destroy_process_group()

if __name__ == '__main__':
    setup()
    


'''
training:
torchrun --nproc_per_node=8 train.py --config-path configs/final-vitl/conv4-dpt2/ load=checkpoints/depth_anything_v2_vitl.pth training.wandb=true

evaluation on testing sets:
torchrun train.py --config-path configs/final-vits/conv32/ inference=true eval.test_datasets=[unreal4k,eth3d]

metrics:
cd evaluation/testdata/
python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/unreal4k/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/unreal4k/teacher504_dpt1.json \
--methods teacher504_dpt1 \
--paths /root/gene/video-depth/configs/hybrid/teacher504_dpt1/test/unreal4k/

python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/unreal4k/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/unreal4k/previous_teacher490_dpt1.json \
--methods previous_teacher490_dpt1 \
--paths /root/gene/video-depth/configs/third-distill/skip123-block4/teacher490_dpt1/unreal4k/

random input:

'''


'''
torchrun train.py --config-path configs/hybrid/teacher490_dpt1 inference=true eval.test_datasets=[unreal4k,eth3d] && cd evaluation/testdata/ && python metrics.py --src_base /root/gene/video-depth/evaluation/testdata/unreal4k/scenes  --output_path /root/gene/video-depth/evaluation/testdata/unreal4k/teacher490_dpt1.json --methods teacher490_dpt1 --paths /root/gene/video-depth/configs/hybrid/teacher490_dpt1/test/unreal4k/; python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/eth3d/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/eth3d/teacher490_dpt1.json \
--methods teacher490_dpt1 \
--paths /root/gene/video-depth/configs/hybrid/teacher490_dpt1/test/eth3d/
'''


'''

torchrun train.py --config-path configs/third-distill/skip123-block4 inference=true hybrid_configs.teacher_resolution=490 eval.test_datasets=[sintel] model.mamba_in_dpt_layer=[1] eval.outfolder='teacher490_dpt1' && cd evaluation/testdata/ && python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/sintel/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/sintel/previous_teacher490_dpt1.json \
--methods previous_teacher490_dpt1 \
--paths /root/gene/video-depth/configs/third-distill/skip123-block4/teacher490_dpt1/sintel/; cd /root/gene/video-depth; \


torchrun train.py --config-path configs/third-distill/skip123-block4 inference=true hybrid_configs.teacher_resolution=490 eval.test_datasets=[waymo] model.mamba_in_dpt_layer=[3] eval.outfolder='teacher490_dpt3' && cd evaluation/testdata/ && python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/waymo/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/waymo/previous_teacher490_dpt3.json \
--methods previous_teacher490_dpt3 \
--paths /root/gene/video-depth/configs/third-distill/skip123-block4/teacher490_dpt3/waymo/; cd /root/gene/video-depth

torchrun train.py --config-path configs/third-distill/skip123-block4 inference=true hybrid_configs.teacher_resolution=518 eval.test_datasets=[waymo] model.mamba_in_dpt_layer=[3] eval.outfolder='teacher518_dpt3'

cd evaluation/testdata/ && python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/unreal4k/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/unreal4k/transformer_rnn.json \
--methods transformer_rnn \
--paths /root/gene/video-depth/configs/transformer_rnn/test/unreal4k/


torchrun train.py --config-path configs/third-distill/skip123-block4 inference=true hybrid_configs.teacher_resolution=490 eval.test_datasets=[unreal4k] model.mamba_in_dpt_layer=[1] eval.outfolder='teacher490_dpt1' eval.test_dataset_resolution=base \
&& cd evaluation/testdata/ && python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/unreal4k/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/unreal4k/previous_teacher490_dpt1_baseres.json \
--methods previous_teacher490_dpt1 \
--paths /root/gene/video-depth/configs/third-distill/skip123-block4/teacher490_dpt1/unreal4k/



torchrun train.py --config-path configs/third-distill/skip123-block4 inference=true hybrid_configs.teacher_resolution=490 eval.test_datasets=[eth3d] model.mamba_in_dpt_layer=[1] eval.outfolder='teacher490_dpt1' && cd evaluation/testdata/ && python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/eth3d/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/eth3d/previous_teacher490_dpt1_no_mamba.json \
--methods previous_teacher490_dpt1 \
--paths /root/gene/video-depth/configs/third-distill/skip123-block4/teacher490_dpt1/eth3d/; cd /root/gene/video-depth; \
torchrun train.py --config-path configs/third-distill/skip123-block4 inference=true hybrid_configs.teacher_resolution=518 eval.test_datasets=[eth3d] model.mamba_in_dpt_layer=[1] eval.outfolder='teacher518_dpt1' && cd evaluation/testdata/ && python metrics.py \
--src_base /root/gene/video-depth/evaluation/testdata/eth3d/scenes  \
--output_path /root/gene/video-depth/evaluation/testdata/eth3d/previous_teacher518_dpt1.json \
--methods previous_teacher518_dpt1 \
--paths /root/gene/video-depth/configs/third-distill/skip123-block4/teacher518_dpt1/eth3d/


'''
