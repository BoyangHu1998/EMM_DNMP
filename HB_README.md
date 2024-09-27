# Reproduce the code with kitti

## 1. preprocess_kitti.py: (preprocess_kitti.sh)
inputs:  
- poses_fn = f'{root_dir}/data_poses/{DRIVE}/poses.txt'
- intrinsic_fn = f'{root_dir}/calibration/perspective.txt'
- cam2pose_fn = f'{root_dir}/calibration/calib_cam_to_pose.txt' 

outputs:  
- center_point.npy
- /calibration
- /data_poses
- raw data of images
<!-- 2. run_colmap_kitti.sh: train the geometry using COLMAP
inputs:
- SEQUENCE="seq_1"
- PROJECT_ROOT='/home/hb/Documents/DNMP-main/colmap_project'
- ROOT_DIR="/home/hb/Documents/DNMP-main/data"
outputs: 
- depth.npy
- normal.npy -->    

## 2. preprocess/run_metashape_kitti.sh:  
inputs:
- SEQUENCE='seq_1'
- PROJECT_ROOT='/home/hb/Documents/DNMP-main'
- ROOT_DIR='/home/hb/Documents/DNMP-main/data'

outputs:
- cameras.xml

## 3. train_kitti360_geo.sh
inputs: pts = pts_data['pts'] where pts_file = "data/seq_1/pcd.npz"


ouputs:  
-{checkpoint}.pth


## 4. train_kitti360_render.sh

