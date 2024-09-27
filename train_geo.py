# This file is used to train the geometry model
from core.config import get_config
from core.trainer_geometry import TrainerGeometry
from core.train_utils import setup_seed
import open3d as o3d

if __name__ == '__main__':

    config = get_config()

    setup_seed(0)
    trainer = TrainerGeometry(config)
    '''
        Debug
    '''
    # voxel_centers = trainer.render.voxel_centers.detach().cpu().numpy()
    # pts = trainer.pts
    # voxel_size = trainer.render.voxel_size
    # # visualize the pts using green , visualize the voxel centers using red
    # pts_pcd = o3d.geometry.PointCloud()
    # pts_pcd.points = o3d.utility.Vector3dVector(pts)
    # pts_pcd.paint_uniform_color([0, 1, 0])
    # o3d.io.write_point_cloud("debug/pts_pcd.ply", pts_pcd)
    # voxel_centers_pcd = o3d.geometry.PointCloud()
    # voxel_centers_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    # voxel_centers_pcd.paint_uniform_color([1, 0, 0])
    # o3d.io.write_point_cloud("debug/voxel_center0001.ply", voxel_centers_pcd)
    # o3d.visualization.draw_geometries([pts_pcd, voxel_centers_pcd])
    # save the pcd

    trainer.train()