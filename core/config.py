import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='DNMP')

# Dataset
parser.add_argument('--dataset', type=str, default='kitti360', help='kitti360, waymo, DTU')
parser.add_argument('--pts_file', type=str, default=None, help='point cloud file')
parser.add_argument('--img_h', type=int, default=376)
parser.add_argument('--img_w', type=int, default=1408)
parser.add_argument('--train_img_path', type=str, default=None)
parser.add_argument('--test_img_path', type=str, default=None)
parser.add_argument('--train_poses_path', type=str, default=None)
parser.add_argument('--test_poses_path', type=str, default=None)
parser.add_argument('--calib_path', type=str, default=None)
parser.add_argument('--cam_to_pose_path', type=str, default=None, help='camera to pose path for KITTI-360 dataset')
parser.add_argument('--depth_path', type=str, default=None, help='depth map dir from COLMAP, saved as .npy files')
parser.add_argument('--normal_path', type=str, default=None, help='normal map dir from COLMAP, saved as .npy files')
parser.add_argument('--center_point_fn', type=str, default=None, help='pre-defined center point pose') # CHECK HERE!
## DTU
parser.add_argument('--dataroot', type=str, default=None, help="data path for DTU dataset")
parser.add_argument('--scence_name', type=str, default=None, help="DTU scence name")
# parser.add_argument('--idx', type=int, help="This scence of DTU dataset collection index?")
parser.add_argument('--idx_imageset', type=int, default=0, help="This scence of DTU dataset collection index?")
parser.add_argument('--num_srcs', type=int, default=5, help="default 5?")
parser.add_argument('--resize_ratio', type=float, default=1.0, help="resize ratio for DTU dataset")


# Train
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--num_rays', type=int, default=1024)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=10000)
parser.add_argument('--chunk_size', type=int, default=2048)
parser.add_argument('--checkpoint_dir', type=str, default=None)
parser.add_argument('--valid_depth_thresh', type=float, default=50.)
parser.add_argument('--use_photo_check', type=str2bool, default=False)
parser.add_argument('--use_photo_loss', type=str2bool, default=False)
parser.add_argument('--use_normal_loss', type=str2bool, default=False)
parser.add_argument('--use_deform_loss', type=str2bool, default=False)
parser.add_argument('--depth_loss_weight', type=float, default=0.05)
parser.add_argument('--normal_loss_weight', type=float, default=0.0)
parser.add_argument('--use_rasterizer_cache', type=str2bool, default=True)
parser.add_argument('--use_depth', type=str2bool, default=False)


# Model
parser.add_argument('--voxel_size', type=float, default=1.0)
parser.add_argument('--max_hits', type=int, default=8, help='AABB test')
parser.add_argument('--voxel_depth_range_thresh', type=float, default=100.)
parser.add_argument('--vertex_feats_dim', type=int, default=32)
parser.add_argument('--near_plane', type=float, default=0.5)
parser.add_argument('--far_plane', type=float, default=150.)
parser.add_argument('--N_freqs_xyz', type=int, default=10)
parser.add_argument('--N_freqs_dir', type=int, default=4)
parser.add_argument('--N_freqs_feats', type=int, default=6)
parser.add_argument('--perturb', type=float, default=0.1)
parser.add_argument('--logscale', type=str2bool, default=False)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--nerf_samples_coarse', type=int, default=128)
parser.add_argument('--nerf_samples_fine', type=int, default=128)
parser.add_argument('--num_near_imgs', type=int, default=3)
parser.add_argument('--pretrained_mesh_ae', type=str, default=None)
parser.add_argument('--mesh_ae_hidden_size', type=int, default=8)
parser.add_argument('--pretrained_geo', type=str, default=None)
parser.add_argument('--use_colmap_normal', type=str2bool, default=False)
parser.add_argument('--render_semantic', type=str2bool, default=False)
parser.add_argument('--pretrained_geo_list', nargs='+', default=[])
parser.add_argument('--voxel_size_list', nargs='+', default=[])
parser.add_argument('--scene_scale', type=float, default=1.0)
parser.add_argument('--use_disp', type=str2bool, default=False)
parser.add_argument('--nerf_net_depth', type=int, default=8)
parser.add_argument('--nerf_net_width', type=int, default=256)
parser.add_argument('--nerf_scale', type=float, default=1.)
parser.add_argument('--use_bkgd', type=str2bool, default=False)
parser.add_argument('--render_multi_scale', type=str2bool, default=False)
parser.add_argument('--max_faces_per_bin_factor', type=int, default=0, help='max faces per bin factor for coarse-to-fine rasterization')
parser.add_argument('--bin_size', type=int, default=0, help='bin_size for coarse-to-fine rasterization')
parser.add_argument('--num_faces', type=int, default=0, help='number of intersected faces')
parser.add_argument('--coarse_num_faces', type=int, default=0, help='number of intersected faces')
parser.add_argument('--use_voxel_center', type=str2bool, default=False) # CHECK HERE!
parser.add_argument('--use_xyz_pos', type=str2bool, default=False)
parser.add_argument('--use_viewdirs', type=str2bool, default=True)
parser.add_argument('--mesh_net_width', type=int, default=256)
parser.add_argument('--mesh_net_depth', type=int, default=8)
parser.add_argument('--mesh_chunk_size', type=str2bool, default=65536)


# Infer
parser.add_argument('--pretrained_render', type=str, default=None)
# parser.add_argument('--pretrained_mipnerf', type=str, default=None) # remove mipnerf
parser.add_argument('--save_dir', type=str, default=None)

parser.add_argument('--mesh_paths', nargs='+', default=[])


def get_config():
    import sys
    print(sys.argv) 
    args = parser.parse_args()
    return args