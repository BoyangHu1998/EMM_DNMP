export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export DATASET="dtu"
export DRIVE="2013_05_28_drive_0000_sync" # TODO
export DATA_ROOT="/home/hb/Documents/DNMP-main/dtu_data" # TODO
export VOXEL_SIZE=0.5 # 1.0
# export VOXEL_SIZE=1.0 # 1.0
export LOG_DIR="training_outputs/log" # TODO 
export CKPT_DIR="training_outputs/chpt" # TODO
export PRETRAINED_MESH_AE="pretrained/mesh_ae/mesh_ae.pth"

python train_geo.py \
--dataset ${DATASET} \
--train_img_path ${DATA_ROOT}/${SEQUENCE}/train_imgs \
--test_img_path ${DATA_ROOT}/${SEQUENCE}/test_imgs \
--train_poses_path ${DATA_ROOT}/data_poses/${DRIVE}/poses.txt \
--test_poses_path ${DATA_ROOT}/data_poses/${DRIVE}/poses.txt \
--pts_file ${DATA_ROOT}/${SEQUENCE}/pcd.npz \
--calib_path ${DATA_ROOT}/calibration/perspective.txt \
--cam_to_pose_path ${DATA_ROOT}/calibration/calib_cam_to_pose.txt \
--log_dir ${LOG_DIR}/geo-${DATASET}-${SEQUENCE}-${VOXEL_SIZE}-${TIME} \
--checkpoint_dir ${CKPT_DIR}/geo-${DATASET}-${SEQUENCE}-${VOXEL_SIZE}-${TIME} \
--num_rays 32768 \
--chunk_size 32768 \
--num_faces 1 \
--voxel_size ${VOXEL_SIZE} \
--print_freq 100 \
--val_freq 5000 \
--N_freqs_xyz 10 \
--N_freqs_dir 4 \
--logscale False \
--scale_factor 1. \
--max_iter 50000 \
--pretrained_mesh_ae ${PRETRAINED_MESH_AE} \
--mesh_ae_hidden_size 8 \
--near_plane 0.5 \
--far_plane 100. \
--center_point_fn ${DATA_ROOT}/${SEQUENCE}/center_point.npy \
--scene_scale 1. \
--use_disp False \
--valid_depth_thresh 50. \
--bin_size 72 \
--depth_path ${DATA_ROOT}/${SEQUENCE}/depth \
--normal_path ${DATA_ROOT}/${SEQUENCE}/normal