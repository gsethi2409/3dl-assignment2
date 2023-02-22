import torch
import pytorch3d.loss
import torch.nn as nn

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grids

	m = nn.Sigmoid()
	loss = nn.BCELoss()
	output = loss(m(voxel_src), voxel_tgt)
	return output

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  

	src_dists, src_idxs, src_nn = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, K=1)
	tgt_dists, tgt_idxs, tgt_nn = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, K=1)

	loss_chamfer = torch.sum(src_dists + tgt_dists)

	return loss_chamfer
	
	# loss_src_tgt, _, _ = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt, None, None, norm = 2, K=1)
	# loss_tgt_src, _, _ = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src, None, None, norm = 2, K=1)

	# loss_chamfer = torch.sum(loss_src_tgt) + torch.sum(loss_tgt_src)

	# return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh_src) 
	# implement laplacian smoothening loss
	
	return loss_laplacian