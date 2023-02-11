import torch
import pytorch3d
import torch.nn as nn

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids

	m = nn.Sigmoid()
	loss = nn.BCELoss()
	output = loss(m(voxel_src), voxel_tgt)
	print(output.shape)
	print(output)

	return output

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	
	loss_src_tgt, _, _ = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, None, None, norm = 2, K=1)
	loss_tgt_src, _, _ = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, None, None, norm = 2, K=1)

	# print('loss_tgt_src: ', loss_tgt_src.shape)
	
	loss_chamfer = torch.sum(loss_src_tgt) + torch.sum(loss_tgt_src)

	# loss_chamfer
	# implement chamfer loss from scratch
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = None 
	# implement laplacian smoothening loss
	return loss_laplacian