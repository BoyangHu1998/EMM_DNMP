import torch
from pytorch3d.io import save_obj
import trimesh
from pytorch3d.renderer import RasterizationSettings, TexturesVertex, MeshRasterizer
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_scene
import numpy as np


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.config import get_config
from models.mesh import DNMPScene
from models.mesh_AE import PointDecoder

pts_file = "/home/hb/Documents/DNMP-hb/dtu_data/dtu_pcd.npz"


def load_chkpt(chkpt_path):
    state = torch.load(chkpt_path)
    return state['global_step'], state['state_dict'], state['optimizer'], state['scheduler'], state['config'], state['decoder_state_dict']
    
def load_mesh(geo_dict):
    mesh_embeddings = geo_dict['mesh.mesh_embeddings'].to(device).data 
    mesh_embeddings = mesh_embeddings / (torch.norm(mesh_embeddings, dim=1, keepdim=True) + 1e-8) # latent vectors of mesh vertices # (mesh_nums, 8)
    norms = torch.norm(mesh_embeddings, dim=1) # (mesh_nums, )
    return mesh_embeddings, norms


def _save_mesh_obj(dnmp_scene, output_path):
    meshes = dnmp_scene.meshes

    # Assuming meshes.verts_list() returns a list of PyTorch tensors
    verts_list = meshes.verts_list()
    verts = torch.cat(verts_list, dim=0)

    # Assuming meshes.faces_list() returns a list of PyTorch tensors
    faces_list = meshes.faces_list()
    faces = torch.cat(faces_list, dim=0)

    save_obj(output_path, verts=verts, faces=faces)

def get_meshes(dnmp_scene):
    # # Load the point cloud data
    # pts_data = np.load(pts_file)
    # # pts = pts_data['pts']
    # colors = pts_data['colors']
    # colors = torch.from_numpy(colors).to(device)
    
    # get the mesh
    meshes = dnmp_scene.meshes
    
    vertex_embeddings = dnmp_scene.vertex_embeddings.view(dnmp_scene.num_meshes, dnmp_scene.num_verts_per_mesh, -1)
    
    
    colors = meshes.vertex_colors
    colors = colors.view(dnmp_scene.num_meshes, dnmp_scene.num_verts_per_mesh, -1)
    
    # put the color to the vertex_embeddings
    vertex_embeddings = torch.cat([vertex_embeddings, colors.unsqueeze(1)], dim=-1)
    
    textures = TexturesVertex(vertex_embeddings[:]) # sampled_idx as all to visualize
    meshes = Meshes(dnmp_scene.verts[:], dnmp_scene.faces[:], textures)
    meshes = join_meshes_as_scene(meshes)
    # verts_features_packed = meshes.textures.verts_features_packed() # ? Unused
    
    return meshes

def save_mesh(dnmp_scene, output_path):
    meshes = get_meshes(dnmp_scene)
    
    # Assuming meshes.verts_list() and meshes.faces_list() return the vertices and faces
    verts = torch.cat(meshes.verts_list(), dim=0).cpu().numpy()
    faces = torch.cat(meshes.faces_list(), dim=0).cpu().numpy()

    # Assuming meshes.textures contains the vertex colors
    colors = torch.cat(meshes.textures.verts_features_list(), dim=0).cpu().detach().numpy()
    # print(colors.shape, colors.min(), colors.max())
    # print(colors)

    # Create a Trimesh mesh object
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)

    # Save the mesh as a PLY file
    mesh.export(output_path, file_type="ply")


def get_decoder_fn(pretrained_geo):
    # Load pretrained mesh auto-encoder
    assert config.pretrained_mesh_ae is not None
    decoder_fn = PointDecoder(config.mesh_ae_hidden_size, voxel_size=1.).to(device)
    # decoder_dict = torch.load(config.pretrained_mesh_ae)
    # decoder_fn.load_state_dict(decoder_dict['decoder_state_dict'])
    
    geo_dict = torch.load(pretrained_geo)['decoder_state_dict']

    decoder_fn.load_state_dict(geo_dict)
    decoder_fn.requires_grad = False
    return decoder_fn

def read_geo(device):
    # Load pretrained geometry
    mesh_embeddings_list = []
    voxel_centers_list = []
    coarse_mesh, fine_mesh = None, None # DNMPScene
        
    # load the geometry data from checkpoint
    for pretrained_geo in config.pretrained_geo_list:
        geo_dict = torch.load(pretrained_geo)['state_dict']

        mesh_embeddings = geo_dict['mesh.mesh_embeddings'].to(device).data
        mesh_embeddings = mesh_embeddings / (torch.norm(mesh_embeddings, dim=1, keepdim=True) + 1e-8)
        norms = torch.norm(mesh_embeddings, dim=1)

        voxel_centers = geo_dict['mesh.voxel_centers'].data
        voxel_centers_offset = geo_dict['mesh.voxel_centers_offset'].data
        voxel_centers = (voxel_centers + voxel_centers_offset).to(device)
        voxel_centers = voxel_centers[~torch.isnan(norms)]
        voxel_centers = voxel_centers / config.scene_scale
        mesh_embeddings = mesh_embeddings[~torch.isnan(norms)]
        mesh_embeddings_list.append(mesh_embeddings)
        voxel_centers_list.append(voxel_centers)
        

    mesh_embeddings = mesh_embeddings_list
    voxel_centers = voxel_centers_list
    # print(mesh_embeddings[0].shape, mesh_embeddings[1].shape)

    voxel_size = [float(v) / config.scene_scale for v in config.voxel_size_list]
    
    decoder_fn = get_decoder_fn(config.pretrained_geo_list[0])
    
    # for i, geo_chkpt_path in enumerate(config.pretrained_geo_list):
        # global_step, geo_dict, optimizer, scheduler, config = load_chkpt(geo_chkpt_path)
        # mesh_embeddings, norms = load_mesh(geo_dict) # latent_code of vertices of meshs
        # meshs = decode_mesh_embedding(mesh_embeddings)
        # save_mesh(meshs, mesh_paths[i])

    # 1.0 voxel_size
    coarse_mesh = DNMPScene(
                            config,
                            voxel_centers=voxel_centers[1],
                            voxel_size=voxel_size[1],
                            mesh_embeddings=mesh_embeddings[1],
                            device=device,
                            num_faces=config.coarse_num_faces,
                            decoder_fn=decoder_fn,
                            vertex_embedding_dim=config.vertex_feats_dim).to(device)
    
    # 0.5 voxel_size
    fine_mesh = DNMPScene(
                config,
                voxel_centers=voxel_centers[0],
                voxel_size=voxel_size[0],
                mesh_embeddings=mesh_embeddings[0],
                device=device,
                num_faces=config.num_faces,
                decoder_fn=decoder_fn,
                vertex_embedding_dim=config.vertex_feats_dim)
    
    vertex_feats_dim = coarse_mesh.vertex_embeddings.shape[-1]
    
    save_mesh(fine_mesh,   config.mesh_paths[0])
    save_mesh(coarse_mesh, config.mesh_paths[1])


if __name__ == '__main__':
    config = get_config()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    assert len(config.pretrained_geo_list) == len(config.mesh_paths) == len(config.voxel_size_list)
    read_geo(device)