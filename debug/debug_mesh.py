import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.config import get_config
from core.tester import Tester
from core.train_utils import setup_seed
import open3d as o3d
import numpy as np
if __name__ == '__main__':

    config = get_config()

    setup_seed(0)
    tester = Tester(config)
    coarse_mesh_verts = tester.render.coarse_mesh.verts.detach().cpu().numpy()
    fine_mesh_verts = tester.render.fine_mesh.verts.detach().cpu().numpy()      
    print(fine_mesh_verts.shape)
    print(coarse_mesh_verts.shape)
    # save the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coarse_mesh_verts[:, 0, :])
    o3d.io.write_point_cloud("./debug/coarse_mesh.ply", pcd)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(fine_mesh_verts[:, 0, :])
    o3d.io.write_point_cloud("./debug/fine_mesh.ply", pcd2)
    
    # np.savez("./debug/coarse_mesh.npz", coarse_mesh_verts=coarse_mesh_verts)
    # np.savez("./debug/fine_mesh.npz", fine_mesh_verts=fine_mesh_verts)
