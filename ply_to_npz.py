import numpy as np
from plyfile import PlyData, PlyElement

def ply_to_npz(ply_filename, npz_filename, save_colors=False):
    ply_data = PlyData.read(ply_filename)

    # pts_data['pts']
    vertices = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T

    if save_colors and 'red' in ply_data['vertex'] and 'green' in ply_data['vertex'] and 'blue' in ply_data['vertex']:
        colors = np.vstack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']]).T
    else:
        print("No colors found in ply file, not saving colors.")
        colors = None
        
    if save_colors and 'red' in ply_data['vertex'] and 'green' in ply_data['vertex'] and 'blue' in ply_data['vertex']:
        colors = np.vstack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']]).T
    else:
        colors = None

    np.savez(npz_filename, pts=vertices, colors=colors)

ply_filename = "data/seq_1/pcd.ply"
npz_filename = "data/seq_1/pcd.npz"

ply_to_npz(ply_filename, npz_filename, save_colors=True)
