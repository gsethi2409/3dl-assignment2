# Single View to 3D

Goals: Here I explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## Exploring loss functions

### 1.1 Fitting a voxel grid 

`python fit_data.py --type 'vox'`

### 1.2 Fitting a point cloud 

`python fit_data.py --type 'point'`

### 1.3 Fitting a mesh

`python fit_data.py --type 'mesh'`

## Reconstructing 3D from single view

### 2.1 Fitting a voxel grid 

`python train_model.py --type 'vox' --batch_size 64 --num_workers 8`

### 2.2 Fitting a pointcloud

`python train_model.py --type 'point' --batch_size 64 --num_workers 8`

### 2.3 Fitting a mesh

`python train_model.py --type 'mesh' --batch_size 64 --num_workers 8`

