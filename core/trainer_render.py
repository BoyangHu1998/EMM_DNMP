import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from datasets import dataset_dict

from models.renderer import DNMPRender
from models.mesh_AE import PointDecoder
from models.ray_utils import get_rays, sample_rays

from core.cameras import get_camera
from core.train_utils import AverageMeter, ensure_dir, get_logger
import core.metrics as metrics

from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
import json
import shutil
import warnings
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`.")
# import matplotlib.pyplot as plt
import cv2

# hb: complete the mesh
# def get_boundaries(hole_mask):                    
#     # 1. get inner hole bourdaries mask from the mask
#     def get_hole_inner_boundaries(hole_mask):
#         hole_boundaries = torch.zeros_like(hole_mask, dtype=torch.int8).to(self.device)
#         hole_boundaries[1:] += hole_mask[:-1]
#         hole_boundaries[:-1] += hole_mask[1:]
#         hole_boundaries[:, 1:] += hole_mask[:, :-1]
#         hole_boundaries[:, :-1] += hole_mask[:, 1:]
#         hole_boundaries = hole_boundaries > 0
#         hole_boundaries = hole_boundaries & ~hole_mask
#         return hole_boundaries
    
#     # 2. get outer hole bourdaries mask from the mask
#     def get_hole_outer_boundaries(hole_mask, has_depth_mask):
#         # Ensure the hole_mask is a float tensor and move it to the same device as 'self.device'
#         hole_mask = hole_mask.float().to(self.device)

#         # Create a dilation kernel. Note: The kernel should be of shape (out_channels, in_channels, height, width)
#         # For binary mask dilation, a single-channel kernel is used. Here, the kernel size is 3x3.
#         kernel_size = 3
#         kernel = torch.ones(1, 1, kernel_size, kernel_size).to(self.device)

#         # Pad the hole_mask to ensure the output size is the same as the input size after convolution
#         padding = kernel_size // 2

#         # Perform the 'dilation' via a convolution operation. The stride is 1 and padding ensures output size matches the input.
#         # We use F.conv2d for the convolution, requiring the input tensor to have a shape of (batch_size, channels, height, width),
#         # hence, we unsqueeze twice to add the required batch_size and channels dimensions.
#         hole_mask_dilated = F.conv2d(hole_mask.unsqueeze(0).unsqueeze(0), kernel, padding=padding, stride=1)

#         # Since we're simulating dilation, we're interested in any overlap. Thus, we threshold the result so that any value
#         # greater than 0 becomes 1, simulating the behavior of dilation where if any part of the kernel overlaps a '1' in the input,
#         # the output is set to '1'.
#         hole_mask_dilated = hole_mask_dilated > 0

#         # Squeeze to remove the batch and channel dimensions added for convolution
#         hole_mask_dilated = hole_mask_dilated.squeeze()

#         # Perform the bitwise AND with has_depth_mask. Ensure has_depth_mask is also a boolean tensor on the same device.
#         has_depth_mask = has_depth_mask.to(self.device)
#         boundaries = hole_mask_dilated & has_depth_mask

#         return boundaries

#     depth = sample['depth'].squeeze().to(self.device)

#     hole_inner_boundaries = get_hole_inner_boundaries(hole_mask)
#     hole_outer_boundaries = get_hole_outer_boundaries(hole_mask, depth!=0)
    
#     # Debug: visualize the hole_boundaries and hole_mask
#     def visualize_hole_boundaries(hole_boundaries, inner=True):
#         # Create a new figure window
#         H, W = hole_boundaries.shape
#         fig = np.zeros((H, W, 3), dtype=np.uint8)

#         # Visualize hole_boundaries
#         hole_boundaries_np = hole_boundaries.detach().cpu().numpy().astype(np.uint8) * 255
#         fig[0:H, 0:W] = cv2.cvtColor(hole_boundaries_np, cv2.COLOR_GRAY2BGR)


#         # Save the figure
#         if inner:
#             cv2.putText(fig, "Inner Hole Boundaries", (int(0.2*W), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.imwrite("hole_inner_boundaries.png", fig)
#         else:
#             cv2.putText(fig, "Outer Hole Boundaries", (int(0.2*W), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.imwrite("hole_outer_boundaries.png", fig)
    
#     # Call the method
#     DEBUG_HOLE_BOUNDARIES = False
#     if DEBUG_HOLE_BOUNDARIES:
#         visualize_hole_boundaries(hole_inner_boundaries.squeeze(), inner=True)
#         visualize_hole_boundaries(hole_outer_boundaries.squeeze(), inner=False)
    
#     return hole_inner_boundaries, hole_outer_boundaries    

# hb: identify the hole use the mask and depth
def identify_hole(sample=None):
    mask = sample['mask'].squeeze()
    depth = sample['depth'].squeeze()
    hole_mask = mask * (depth == 0)
    
    # hb: visualize the mask, depth ==0, hole_mask in a single image 
    def visualize_mask_depth_hole_mask(mask, depth, hole_mask):
        # Create a new figure window
        H, W = mask.shape
        fig = np.zeros((H, W * 3, 3), dtype=np.uint8)

        # Visualize mask
        mask_np = mask.detach().cpu().numpy().astype(np.uint8) * 255
        fig[0:H, 0:W] = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        cv2.putText(fig, "Mask", (int(0.2*W), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Visualize depth == 0
        # depth_zero_resized = cv2.resize((depth == 0).astype(np.uint8) * 255, (400, 400))
        depth_zero_resized = (depth == 0).detach().cpu().numpy().astype(np.uint8) * 255
        fig[0:H, W:2*W] = cv2.cvtColor(depth_zero_resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(fig, "Depth == 0", (int(1.2*W), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Visualize hole_mask
        # hole_mask_resized = cv2.resize(hole_mask, (400, 400))
        hole_mask_np = hole_mask.detach().cpu().numpy().astype(np.uint8) * 255
        fig[0:H, 2*W:3*W] = cv2.cvtColor(hole_mask_np, cv2.COLOR_GRAY2BGR)
        cv2.putText(fig, "Hole Mask", (int(2.2*W), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save the figure
        cv2.imwrite("mask_depth_hole_mask.png", fig)

    # Call the method
    DEBUG_MASK = False
    if DEBUG_MASK:
        visualize_mask_depth_hole_mask(mask.squeeze(), depth.squeeze(), hole_mask.squeeze())
    
    return hole_mask, sum(hole_mask.view(-1))            

class TrainerRender:

    def __init__(self, config):
        self.config = config
        self.config.near_plane = self.config.near_plane / self.config.scene_scale
        self.config.far_plane = self.config.far_plane / self.config.scene_scale
        
        ensure_dir(self.config.log_dir)
        self.logger = get_logger(os.path.join(self.config.log_dir, 'train.log'))
        self.logger.info(self.config)
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.log_dir, 'tensorboard'))
        self.checkpoint_dir = self.config.checkpoint_dir
        ensure_dir(self.checkpoint_dir)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('GPU not available')
        
        # Recenter the scene using center point
        if self.config.center_point_fn is not None:
            center_point = np.load(self.config.center_point_fn)
            self.transform_M = np.linalg.inv(center_point)
        else:
            self.transform_M = np.eye(4)

        # Load pretrained mesh auto-encoder
        assert self.config.pretrained_mesh_ae is not None
        self.decoder_fn = PointDecoder(config.mesh_ae_hidden_size, voxel_size=1.).to(self.device)
        decoder_dict = torch.load(self.config.pretrained_mesh_ae)
        self.decoder_fn.load_state_dict(decoder_dict['decoder_state_dict'])
        self.decoder_fn.requires_grad = False
        
        # Load pretrained geometry
        assert len(self.config.pretrained_geo_list) > 0 # 1: single scale 2: multi scale (only support 2 scales in the cuurent version)
        assert len(self.config.pretrained_geo_list) == len(self.config.voxel_size_list)
        mesh_embeddings_list = []
        voxel_centers_list = []
        for pretrained_geo in self.config.pretrained_geo_list:

            geo_dict = torch.load(pretrained_geo)['state_dict']

            mesh_embeddings = geo_dict['mesh.mesh_embeddings'].to(self.device).data
            mesh_embeddings = mesh_embeddings / (torch.norm(mesh_embeddings, dim=1, keepdim=True) + 1e-8)
            norms = torch.norm(mesh_embeddings, dim=1)

            voxel_centers = geo_dict['mesh.voxel_centers'].data
            # voxel_centers_offset = geo_dict['mesh.voxel_centers_offset'].data   // hb: remove for voxel_centers_offset
            # voxel_centers = (voxel_centers + voxel_centers_offset).to(self.device)
            voxel_centers = (voxel_centers).to(self.device)
            voxel_centers = voxel_centers[~torch.isnan(norms)]
            voxel_centers = voxel_centers / self.config.scene_scale
            mesh_embeddings = mesh_embeddings[~torch.isnan(norms)]
            mesh_embeddings_list.append(mesh_embeddings)
            voxel_centers_list.append(voxel_centers)

        mesh_embeddings = mesh_embeddings_list
        voxel_centers = voxel_centers_list
        voxel_size = [float(v) / self.config.scene_scale for v in self.config.voxel_size_list]
        
        self.render = DNMPRender(
            self.config, 
            self.device, 
            voxel_size, 
            voxel_centers, 
            mesh_embeddings, 
            self.decoder_fn).to(self.device)
        # load checkpoint from training_outputs/chpt/render-dtu-0.001-0.005-test/ckpt_10000.pth (temporary)
        path = r'training_outputs/chpt/render-dtu-0.001-0.005-test/ckpt_10000.pth'
        self.render.load_state_dict(torch.load(path)['state_dict'])
        
        
        config_dict = vars(self.config)
        config_file = os.path.join(self.config.log_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=1)

        snapshot_dir = os.path.join(self.config.log_dir, 'snapshot')
        ensure_dir(snapshot_dir)

        BASE_DIR = os.getcwd()
        
        # hb: rm for repeatability
        if os.path.exists(os.path.join(snapshot_dir, 'core')):
            shutil.rmtree(os.path.join(snapshot_dir, 'core'), ignore_errors=True)
        if os.path.exists(os.path.join(snapshot_dir, 'models')):
            shutil.rmtree(os.path.join(snapshot_dir, 'models'), ignore_errors=True)
        if os.path.exists(os.path.join(snapshot_dir, 'scripts')):
            shutil.rmtree(os.path.join(snapshot_dir, 'scripts'), ignore_errors=True)
        # copy core files for repeatability            
        shutil.copytree(os.path.join(BASE_DIR, 'models'), \
            os.path.join(os.path.join(snapshot_dir, 'models')))
        shutil.copytree(os.path.join(BASE_DIR, 'core'), \
            os.path.join(os.path.join(snapshot_dir, 'core')))
        shutil.copytree(os.path.join(BASE_DIR, 'scripts'), \
            os.path.join(os.path.join(snapshot_dir, 'scripts')))
        
        
        
        self.optimizer = Adam([{'params': self.render.parameters()}], 
                               lr=self.config.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)
        
        self.scheduler = ExponentialLR(self.optimizer, 0.999999)
       
        self.global_step = 0
        self.val_step = 0

        self.max_iter = self.config.max_iter

        self.get_dataloader()
    
 
    
    def get_dataloader(self):
        dataset = dataset_dict[self.config.dataset]

        self.train_dataset = dataset(self.config, phase='train')
        self.val_dataset = dataset(self.config, phase='val')

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=self.config.num_workers,
                                                   pin_memory=False)

        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.num_workers,
                                                 pin_memory=False)
        
        self.val_iter = self.val_loader.__iter__()
        self.val_iter_num = 0
        self.max_val_iter_num = len(self.val_loader)
    
        

    
    def train(self):
        self.render.train()

        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()

        flag = True
        
        # hd: initialize the sample index for mesh completion
        when_add_mesh = 1000


        while flag:
            for idx, sample in enumerate(self.train_loader):
                
                # start training
                # add_mesh = (self.global_step + 1) % when_add_mesh == 0
                add_mesh = (self.global_step) % when_add_mesh == 0
                # add_mesh = False


                if self.global_step > self.max_iter:
                    flag = False
                    break

                self.global_step += 1
                self.optimizer.zero_grad()

                rgb = sample['rgb'].squeeze()
                pose = sample['pose'].squeeze().numpy()
                intrinsic = sample['intrinsic'].squeeze().numpy()
                img_hw = sample['img_hw'].squeeze().numpy().astype(int)

                if 'depth' in sample.keys() and self.config.use_depth:
                    depth = sample['depth'].squeeze() / self.config.scene_scale
                    depth = depth.to(self.device)

                pose_ = np.eye(4)
                pose_[:3, :4] = pose
                # recenter and rescale poses
                pose = np.matmul(self.transform_M, pose_)
                pose[:3,3] = pose[:3,3] / self.config.scene_scale

                rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

                rays_o = torch.from_numpy(rays_o).float().to(self.device)
                rays_d = torch.from_numpy(rays_d).float().to(self.device)
                if not torch.is_tensor(rgb):
                    rgb = torch.from_numpy(rgb).float()
                rgb = rgb.to(self.device)

                camera = get_camera(intrinsic, pose, img_hw, self.device)
            
                # # hb: 2. preprocess add mesh completion
                if self.config.render_multi_scale:
                    if add_mesh: # mesh completion                    
                        # hb: 1. Preprocess add mesh completion
                        torch.cuda.empty_cache()
                        mask = sample['mask'].to(self.device).view(-1)
                        rays_o_ = rays_o.view(-1, 3)
                        rays_d_ = rays_d.view(-1, 3)
                        pix_coords_idx = torch.arange(0, rays_o.shape[0] * rays_o.shape[1]).long().to(self.device)
                        with torch.no_grad(): # hb: use no_grad to avoid out of memory
                            ret_dict = self.render.multiscale_find_mesh(rays_o_, rays_d_, pix_coords_idx, camera, img_hw=img_hw, mask=mask, voxel_size=self.config.voxel_size_list)
                        
                            # Update the depth of the scene
                            boundary_depth, boundary_idx = ret_dict['new_depth'], ret_dict['inner_boundary_idx']
                            boundary_idx_2d = torch.stack([boundary_idx // img_hw[1], boundary_idx % img_hw[1]], dim=1).long()
                            depth[tuple(boundary_idx_2d.T)] = boundary_depth

                            # hb: 3. Add mesh completion
                            self.render.multiscale_add_mesh(ret_dict, voxel_size=self.config.voxel_size_list)
                            # Reset optimizer and scheduler HERE!
                            self.optimizer_scheduler_reset(self.global_step)
                    
                            
                    # render multi scale mesh
                    rays_o_, rays_d_, rgb_, pix_coords_idx = sample_rays(rays_o, rays_d, rgb, self.config.num_rays)
                    pix_coords_idx = torch.from_numpy(pix_coords_idx).long().to(self.device)
            
                    ret_dict = self.render.render_rays_multiscale(rays_o_, rays_d_, pix_coords_idx, camera, img_hw=img_hw)
                    
                    valid_idx = ret_dict['valid_idx']
                    loss = ((ret_dict['rgb'][...,:3] - rgb_) ** 2)[valid_idx].mean()
                    mse = loss.detach().cpu().numpy()

                    # if use_depth is True, use depth for additional supervision
                    if 'depth' in sample.keys() and self.config.use_depth:
                        depth_ = depth.view(-1)[pix_coords_idx]
                        depth_loss = torch.abs(ret_dict['depth'] - depth_)
                        depth_loss = torch.sum(depth_loss) # hb: add one line to make the depth_lose a scaler, to make the code work 
                        loss = loss + depth_loss * 0.01 * self.config.scene_scale
                    
                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()
                    
                    psnr = metrics.mse2psnr_np(mse)
                    loss_meter.update(loss.item())
                    psnr_meter.update(psnr)

                    if self.global_step % self.config.print_freq == 0:
                        self.logger.info(
                            'Iter [{}/{}] Loss: {:.4f} PSNR: {:.4f}'.format(
                                self.global_step, self.max_iter, loss_meter.avg, psnr_meter.avg))
                        self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                        self.writer.add_scalar('train/psnr', psnr_meter.avg, self.global_step)
                        loss_meter.reset()
                        psnr_meter.reset()


                else:
                    # throw exception
                    raise ValueError('render one scale is not supported in the current version. Please use render_multi_scale=True.')
                

                if self.global_step % self.config.val_freq == 0:
                    self.render.eval()
                    self.validate()
                    self.render.train()
                    self.render.config.perturb = self.config.perturb
                    # self.save_ckpt(f'ckpt_{self.global_step}')
                    self.save_entire_model_ckpt(f'ckpt_{self.global_step}')

                    self.logger.info(f'Checkpoint saved at step {self.global_step}')
                    
                # # hb: 3. postprocess add mesh completion (DEBUG)
                # if add_mesh:
                #     print("Add mesh completion")
                #     self.render.eval()
                #     self.validate()
                #     self.render.train()
                #     self.render.config.perturb = self.config.perturb
                #     self.save_entire_model_ckpt(f'ckpt_{self.global_step}')
                #     self.logger.info(f'Checkpoint saved at step {self.global_step}')
    
    def optimizer_scheduler_reset(self, global_steps):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print('Resetting the optimizer and scheduler...')
        # 1. clear optimizer
        del self.optimizer
        # 2. clear scheduler
        cur_scheduler_state = self.scheduler.state_dict()
        del self.scheduler
        
        # 3. reset optimizer
        self.optimizer = Adam([{'params': self.render.parameters()}], 
                               lr=self.config.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)
        # 4. reset scheduler
        self.scheduler = ExponentialLR(self.optimizer, 0.999999)
        self.scheduler.load_state_dict(cur_scheduler_state)
        # if global_steps > 0:
        #     for _ in range(global_steps):
        #         self.scheduler.step()

    
    def validate(self):

        self.render.config.perturb = 0.0

        with torch.no_grad():
            # only evaluate one image to save time
            # sample = self.val_iter.next()
            sample = next(self.val_iter)
            self.val_iter_num += 1
            if self.val_iter_num == self.max_val_iter_num:
                self.val_iter = self.val_loader.__iter__()
                self.val_iter_num = 0

            rgb = sample['rgb'].to(self.device).squeeze()
            pose = sample['pose'].squeeze().numpy()
            intrinsic = sample['intrinsic'].squeeze().numpy()
            img_hw = sample['img_hw'].squeeze().numpy()
            pose_ = np.eye(4)
            pose_[:3, :4] = pose
            pose = np.matmul(self.transform_M, pose_)
            pose[:3,3] = pose[:3,3] / self.config.scene_scale
            rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)

            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)

            camera = get_camera(intrinsic, pose, img_hw, self.device)
            pix_coords_idx = torch.arange(0, rays_o.shape[0]).long().to(self.device)
            
            if self.config.render_multi_scale:
                ret_dict = self.render.inference_img_multi_scale(rays_o, rays_d, camera, img_hw=img_hw)
            else:
                ret_dict = self.render.inference_img(rays_o, rays_d, camera, img_hw=img_hw)

            pred_rgb = ret_dict['rgb'][...,:3].view(int(img_hw[0]), int(img_hw[1]), 3)
            pred_rgb = pred_rgb.cpu().numpy()
            rgb = rgb.cpu().numpy()

            pred_depth = ret_dict['depth'].view(int(img_hw[0]), int(img_hw[1]))
            pred_depth = pred_depth.cpu().numpy()

            valid_idx = ret_dict['valid_idx'].cpu().numpy()

            mse = ((pred_rgb - rgb) ** 2).reshape(-1, 3)[valid_idx].mean()
            psnr = metrics.mse2psnr_np(mse)
            ssim = metrics.ssim_fn(pred_rgb, rgb)
    
        self.logger.info(f'PSNR: {psnr} SSIM: {ssim}')

        self.writer.add_scalar('val/psnr', psnr, self.val_step)
        self.writer.add_scalar('val/ssim', ssim, self.val_step)

        self.writer.add_image('val/pred_image', pred_rgb, self.val_step, dataformats='HWC')
        self.writer.add_image('val/gt_image', rgb, self.val_step, dataformats='HWC')
        self.writer.add_image('val/pred_depth', pred_depth, self.val_step, dataformats='HW')

        self.val_step += 1
    
    def save_ckpt(self, filename='checkpoint'):

        self.logger.info(f'Save checkpoint for iteration {self.global_step}.')
        state = {            
            'global_step': self.global_step,
            'state_dict': self.render.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        torch.save(state, filename)
    
    # hb: 
    def save_entire_model_ckpt(self, filename='checkpoint'):

        self.logger.info(f'Save checkpoint for iteration {self.global_step}.')
        model = self.render
        filename = os.path.join(self.checkpoint_dir, f'{filename}_entire_model.pth')
        torch.save(model, filename)
        
        
        
        
class TestRender:
    def __init__(self, config, render):
        self.config = config
        self.config.near_plane = self.config.near_plane / self.config.scene_scale
        self.config.far_plane = self.config.far_plane / self.config.scene_scale
        
        # ensure_dir(self.config.log_dir)
        # self.logger = get_logger(os.path.join(self.config.log_dir, 'train.log'))
        # self.logger.info(self.config)
        # self.writer = SummaryWriter(log_dir=os.path.join(self.config.log_dir, 'tensorboard'))
        # self.checkpoint_dir = self.config.checkpoint_dir
        # ensure_dir(self.checkpoint_dir)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('GPU not available')
        
        # Recenter the scene using center point
        if self.config.center_point_fn is not None:
            center_point = np.load(self.config.center_point_fn)
            self.transform_M = np.linalg.inv(center_point)
        else:
            self.transform_M = np.eye(4)

        # Load pretrained mesh auto-encoder
        assert self.config.pretrained_mesh_ae is not None
        self.decoder_fn = PointDecoder(config.mesh_ae_hidden_size, voxel_size=1.).to(self.device)
        decoder_dict = torch.load(self.config.pretrained_mesh_ae)
        self.decoder_fn.load_state_dict(decoder_dict['decoder_state_dict'])
        self.decoder_fn.requires_grad = False
        
        # # Load pretrained geometry
        # assert len(self.config.pretrained_geo_list) > 0 # 1: single scale 2: multi scale (only support 2 scales in the cuurent version)
        # assert len(self.config.pretrained_geo_list) == len(self.config.voxel_size_list)
        # mesh_embeddings_list = []
        # voxel_centers_list = []
        # for pretrained_geo in self.config.pretrained_geo_list:

        #     geo_dict = torch.load(pretrained_geo)['state_dict']

        #     mesh_embeddings = geo_dict['mesh.mesh_embeddings'].to(self.device).data
        #     mesh_embeddings = mesh_embeddings / (torch.norm(mesh_embeddings, dim=1, keepdim=True) + 1e-8)
        #     norms = torch.norm(mesh_embeddings, dim=1)

        #     voxel_centers = geo_dict['mesh.voxel_centers'].data
        #     # voxel_centers_offset = geo_dict['mesh.voxel_centers_offset'].data   // hb: remove for voxel_centers_offset
        #     # voxel_centers = (voxel_centers + voxel_centers_offset).to(self.device)
        #     voxel_centers = (voxel_centers).to(self.device)
        #     voxel_centers = voxel_centers[~torch.isnan(norms)]
        #     voxel_centers = voxel_centers / self.config.scene_scale
        #     mesh_embeddings = mesh_embeddings[~torch.isnan(norms)]
        #     mesh_embeddings_list.append(mesh_embeddings)
        #     voxel_centers_list.append(voxel_centers)

        # mesh_embeddings = mesh_embeddings_list
        # voxel_centers = voxel_centers_list
        voxel_size = [float(v) / self.config.scene_scale for v in self.config.voxel_size_list]
        
        self.render = render.to(self.device)
        
        config_dict = vars(self.config)
        config_file = os.path.join(self.config.log_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=1)

        # snapshot_dir = os.path.join(self.config.log_dir, 'snapshot')
        # ensure_dir(snapshot_dir)

        # BASE_DIR = os.getcwd()
        
        # # hb: rm for repeatability
        # if os.path.exists(os.path.join(snapshot_dir, 'core')):
        #     shutil.rmtree(os.path.join(snapshot_dir, 'core'), ignore_errors=True)
        # if os.path.exists(os.path.join(snapshot_dir, 'models')):
        #     shutil.rmtree(os.path.join(snapshot_dir, 'models'), ignore_errors=True)
        # if os.path.exists(os.path.join(snapshot_dir, 'scripts')):
        #     shutil.rmtree(os.path.join(snapshot_dir, 'scripts'), ignore_errors=True)
        # # copy core files for repeatability            
        # shutil.copytree(os.path.join(BASE_DIR, 'models'), \
        #     os.path.join(os.path.join(snapshot_dir, 'models')))
        # shutil.copytree(os.path.join(BASE_DIR, 'core'), \
        #     os.path.join(os.path.join(snapshot_dir, 'core')))
        # shutil.copytree(os.path.join(BASE_DIR, 'scripts'), \
        #     os.path.join(os.path.join(snapshot_dir, 'scripts')))
        
        # self.optimizer = Adam([{'params': self.render.parameters()}], 
        #                        lr=self.config.lr,
        #                        betas=(0.9, 0.999),
        #                        weight_decay=0.)
        
        # self.scheduler = ExponentialLR(self.optimizer, 0.999999)
       
        self.global_step = 0
        self.val_step = 0

        self.max_iter = self.config.max_iter

        self.get_dataloader()
        
    
    def get_dataloader(self):
        dataset = dataset_dict[self.config.dataset]

        self.train_dataset = dataset(self.config, phase='train')
        self.val_dataset = dataset(self.config, phase='val')

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=self.config.num_workers,
                                                   pin_memory=False)

        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.num_workers,
                                                 pin_memory=False)
        
        self.val_iter = self.val_loader.__iter__()
        self.val_iter_num = 0
        self.max_val_iter_num = len(self.val_loader)
    
        

    
    def train(self):
        self.render.train()

        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()

        flag = True
        
        # hd: initialize the sample index for mesh completion
        when_add_mesh = 1


        while flag:
            for idx, sample in enumerate(self.train_loader):
                
                # start training
                # add_mesh = (self.global_step + 1) % when_add_mesh == 0
                add_mesh = (self.global_step) % when_add_mesh == 0

                if self.global_step > self.max_iter:
                    flag = False
                    break

                self.global_step += 1
                self.optimizer.zero_grad()

                rgb = sample['rgb'].squeeze()
                pose = sample['pose'].squeeze().numpy()
                intrinsic = sample['intrinsic'].squeeze().numpy()
                img_hw = sample['img_hw'].squeeze().numpy().astype(int)

                if 'depth' in sample.keys() and self.config.use_depth:
                    depth = sample['depth'].squeeze() / self.config.scene_scale
                    depth = depth.to(self.device)

                pose_ = np.eye(4)
                pose_[:3, :4] = pose
                # recenter and rescale poses
                pose = np.matmul(self.transform_M, pose_)
                pose[:3,3] = pose[:3,3] / self.config.scene_scale

                rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

                rays_o = torch.from_numpy(rays_o).float().to(self.device)
                rays_d = torch.from_numpy(rays_d).float().to(self.device)
                if not torch.is_tensor(rgb):
                    rgb = torch.from_numpy(rgb).float()
                rgb = rgb.to(self.device)

                camera = get_camera(intrinsic, pose, img_hw, self.device)
            
                # # hb: 2. preprocess add mesh completion
                if self.config.render_multi_scale:
                    if add_mesh: # mesh completion                    
                        # hb: 1. Preprocess add mesh completion
                        torch.cuda.empty_cache()
                        mask = sample['mask'].to(self.device).view(-1)
                        rays_o_ = rays_o.view(-1, 3)
                        rays_d_ = rays_d.view(-1, 3)
                        pix_coords_idx = torch.arange(0, rays_o.shape[0] * rays_o.shape[1]).long().to(self.device)
                        with torch.no_grad(): # hb: use no_grad to avoid out of memory
                            ret_dict = self.render.multiscale_find_mesh(rays_o_, rays_d_, pix_coords_idx, camera, img_hw=img_hw, mask=mask, voxel_size=self.config.voxel_size_list)
                        
                            # Update the depth of the scene
                            boundary_depth, boundary_idx = ret_dict['new_depth'], ret_dict['inner_boundary_idx']
                            boundary_idx_2d = torch.stack([boundary_idx // img_hw[1], boundary_idx % img_hw[1]], dim=1).long()
                            depth[tuple(boundary_idx_2d.T)] = boundary_depth

                            # hb: 3. Add mesh completion
                            self.render.multiscale_add_mesh(ret_dict, voxel_size=self.config.voxel_size_list)
                            # Reset optimizer and scheduler HERE!
                            self.optimizer_scheduler_reset(self.global_step)
                            
                    else: # render multi scale mesh
                        rays_o_, rays_d_, rgb_, pix_coords_idx = sample_rays(rays_o, rays_d, rgb, self.config.num_rays)
                        pix_coords_idx = torch.from_numpy(pix_coords_idx).long().to(self.device)
                
                        ret_dict = self.render.render_rays_multiscale(rays_o_, rays_d_, pix_coords_idx, camera, img_hw=img_hw)
                        
                        valid_idx = ret_dict['valid_idx']
                        loss = ((ret_dict['rgb'][...,:3] - rgb_) ** 2)[valid_idx].mean()
                        mse = loss.detach().cpu().numpy()

                        # if use_depth is True, use depth for additional supervision
                        if 'depth' in sample.keys() and self.config.use_depth:
                            depth_ = depth.view(-1)[pix_coords_idx]
                            depth_loss = torch.abs(ret_dict['depth'] - depth_)
                            depth_loss = torch.sum(depth_loss) # hb: add one line to make the depth_lose a scaler, to make the code work 
                            loss = loss + depth_loss * 0.01 * self.config.scene_scale
                        
                        loss.backward()

                        self.optimizer.step()
                        self.scheduler.step()
                        
                        psnr = metrics.mse2psnr_np(mse)
                        loss_meter.update(loss.item())
                        psnr_meter.update(psnr)

                        if self.global_step % self.config.print_freq == 0:
                            self.logger.info(
                                'Iter [{}/{}] Loss: {:.4f} PSNR: {:.4f}'.format(
                                    self.global_step, self.max_iter, loss_meter.avg, psnr_meter.avg))
                            self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                            self.writer.add_scalar('train/psnr', psnr_meter.avg, self.global_step)
                            loss_meter.reset()
                            psnr_meter.reset()


                else:
                    # throw exception
                    raise ValueError('render one scale is not supported in the current version. Please use render_multi_scale=True.')
                

                if self.global_step % self.config.val_freq == 0:
                    self.render.eval()
                    self.validate()
                    self.render.train()
                    self.render.config.perturb = self.config.perturb
                    self.save_ckpt(f'ckpt_{self.global_step}')
                    self.logger.info(f'Checkpoint saved at step {self.global_step}')
                    
                # hb: 3. postprocess add mesh completion (DEBUG)
                if add_mesh:
                    print("Add mesh completion")
                    self.render.eval()
                    self.validate()
                    self.render.train()
                    self.render.config.perturb = self.config.perturb
                    self.save_entire_model_ckpt(f'ckpt_{self.global_step}')
                    self.logger.info(f'Checkpoint saved at step {self.global_step}')
    
    def optimizer_scheduler_reset(self, global_steps):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print('Resetting the optimizer and scheduler...')
        # 1. clear optimizer
        del self.optimizer
        # 2. clear scheduler
        cur_scheduler_state = self.scheduler.state_dict()
        del self.scheduler
        
        # 3. reset optimizer
        self.optimizer = Adam([{'params': self.render.parameters()}], 
                               lr=self.config.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)
        # 4. reset scheduler
        self.scheduler = ExponentialLR(self.optimizer, 0.999999)
        self.scheduler.load_state_dict(cur_scheduler_state)
        # if global_steps > 0:
        #     for _ in range(global_steps):
        #         self.scheduler.step()

    
    def validate(self):

        self.render.config.perturb = 0.0

        with torch.no_grad():
            # only evaluate one image to save time
            # sample = self.val_iter.next()
            sample = next(self.val_iter)
            self.val_iter_num += 1
            if self.val_iter_num == self.max_val_iter_num:
                self.val_iter = self.val_loader.__iter__()
                self.val_iter_num = 0

            rgb = sample['rgb'].to(self.device).squeeze()
            pose = sample['pose'].squeeze().numpy()
            intrinsic = sample['intrinsic'].squeeze().numpy()
            img_hw = sample['img_hw'].squeeze().numpy()
            pose_ = np.eye(4)
            pose_[:3, :4] = pose
            pose = np.matmul(self.transform_M, pose_)
            pose[:3,3] = pose[:3,3] / self.config.scene_scale
            rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)

            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)

            camera = get_camera(intrinsic, pose, img_hw, self.device)
            pix_coords_idx = torch.arange(0, rays_o.shape[0]).long().to(self.device)
            
            if self.config.render_multi_scale:
                ret_dict = self.render.inference_img_multi_scale(rays_o, rays_d, camera, img_hw=img_hw)
            else:
                ret_dict = self.render.inference_img(rays_o, rays_d, camera, img_hw=img_hw)

            pred_rgb = ret_dict['rgb'][...,:3].view(int(img_hw[0]), int(img_hw[1]), 3)
            pred_rgb = pred_rgb.cpu().numpy()
            rgb = rgb.cpu().numpy()

            pred_depth = ret_dict['depth'].view(int(img_hw[0]), int(img_hw[1]))
            pred_depth = pred_depth.cpu().numpy()

            valid_idx = ret_dict['valid_idx'].cpu().numpy()

            mse = ((pred_rgb - rgb) ** 2).reshape(-1, 3)[valid_idx].mean()
            psnr = metrics.mse2psnr_np(mse)
            ssim = metrics.ssim_fn(pred_rgb, rgb)
    
        self.logger.info(f'PSNR: {psnr} SSIM: {ssim}')

        self.writer.add_scalar('val/psnr', psnr, self.val_step)
        self.writer.add_scalar('val/ssim', ssim, self.val_step)

        self.writer.add_image('val/pred_image', pred_rgb, self.val_step, dataformats='HWC')
        self.writer.add_image('val/gt_image', rgb, self.val_step, dataformats='HWC')
        self.writer.add_image('val/pred_depth', pred_depth, self.val_step, dataformats='HW')

        self.val_step += 1