import torch
import imageio
import pytorch3d
import numpy as np

from utils import get_device, get_mesh_renderer, get_points_renderer

import matplotlib.pyplot as plt

def render_turntable_pointcloud(
        pointcloud, image_size=256, device=None,
):

    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=256, background_color=(1, 1, 1)
    )

    verts = torch.Tensor(pointcloud).to(device)
    rgb = torch.rand(pointcloud.shape).to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    
    angles = np.linspace(0, 360)
    pc = []

    for a in angles:
        R, T = pytorch3d.renderer.look_at_view_transform(2, 10, a)
        R = R @ torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        
        rend = rend.cpu().detach().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

        pc.append(rend)

    return pc

def render_turntable_voxelgrid(
    voxels, image_size=256, device=None,
):

    if device is None:
        device = get_device()

    cubemesh = pytorch3d.ops.cubify(voxels, 0.2)
    vertices = torch.tensor(cubemesh.verts_list()[0])
    faces =    torch.tensor(cubemesh.faces_list()[0])
    print(len(vertices[0]))

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    # vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.rand(vertices.shape).to(device)  # (1, N_v, 3)
    # textures = textures * torch.tensor([0.7, 0.7, 1]).to(device) # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    angles = np.linspace(0, 360)
    images = []

    for a in angles:
        pose = pytorch3d.renderer.cameras.look_at_view_transform(3, 0, a)
    
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=pose[0], T=pose[1], fov=60, device=device
        )

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        images.append(rend)

    return images

# if __name__ == "__main__":
#     images = render_turntable_cow(cow_path="data/cow.obj", image_size=512)
#     imageio.mimsave('cow_turntable_render.gif', images, fps=15)
