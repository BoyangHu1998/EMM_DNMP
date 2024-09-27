export CURR_PRETRAIN_GEO_05='/home/hb/Documents/DNMP-main/training_outputs/chpt/geo-kitti360-seq_1-0.5-2023-10-19_05-50-15/ckpt_50000.pth'
export CURR_PRETRAIN_GEO_10='/home/hb/Documents/DNMP-main/training_outputs/chpt/geo-kitti360-seq_1-1.0-2023-09-15_10-41-50/ckpt_50000.pth' 
export COARSE_MESH='/home/hb/Documents/DNMP-main/visualizer-hb/mesh_05.ply'
export FINE_MESH='/home/hb/Documents/DNMP-main/visualizer-hb/mesh_10.ply'

python visualizer_mesh_hb.py \
    --bin_size 72 \
    --max_faces_per_bin_factor 2 \
    --use_rasterizer_cache True \
    --coarse_num_faces 2 \
    --num_faces 4 \
    --pretrained_mesh_ae "/home/hb/Documents/DNMP-main/pretrained/mesh_ae/mesh_ae.pth" \
    --mesh_ae_hidden_size 8 \
    --scene_scale 1.0 \
    --pretrained_geo_list ${CURR_PRETRAIN_GEO_05} ${CURR_PRETRAIN_GEO_10} \
    --voxel_size_list 0.50 1.0 \
    --mesh_paths ${COARSE_MESH} ${FINE_MESH} \
