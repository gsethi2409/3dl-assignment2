from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:

            # self.decoder = nn.ConvTranspose3d(512, 1, kernel_size=32, stride=1)
            # self.decoder = nn.Conv3d(512, 1, kernel_size=32, stride=1, padding=31)
        
            # self.layer1 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(512, 128, kernel_size=8, stride=2, padding=1),
            #     torch.nn.BatchNorm3d(128),
            # )
            # self.layer2 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(128, 32, kernel_size=8, stride=2, padding=1),
            #     torch.nn.BatchNorm3d(32),
            # )
            # self.layer3 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
            #     torch.nn.BatchNorm3d(8),
            # )
            # self.layer4 = torch.nn.Sequential(
            #     torch.nn.ConvTranspose3d(8, 1, kernel_size=1),
            # )

            self.decoder = nn.Sequential(nn.Flatten(),
                                 nn.Linear(512, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048, 8192),
                                 nn.ReLU(),
                                 nn.Linear(8192, 32768))
        
        
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points

            self.decoder = nn.Linear(512, (args.n_points * 3))
            # self.decoder = nn.Sequential(
            #     nn.Linear(512, (args.n_points * 3))
            # )  


            # self.decoder = nn.Sequential(nn.Flatten(),
            #                     nn.Linear(512, 1024),
            #                     nn.ReLU(),
            #                     nn.Linear(1024, 4096),
            #                     nn.ReLU(),
            #                     nn.Linear(4096, self.n_point * 3))

        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.decoder = nn.Linear(512, (mesh_pred.verts_packed().shape[0] * 3))


            # self.decoder = nn.Sequential(nn.Flatten(),
            #                     nn.Linear(512, 1024),
            #                     nn.ReLU(),
            #                     nn.Linear(1024, 4096),
            #                     nn.ReLU(),
            #                     nn.Linear(4096, mesh_pred.verts_packed().shape[0] * 3))


    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            print('encoded_feat.shape: ', encoded_feat.shape)
            encoded_feat = encoded_feat.unsqueeze(-1)
            print('encoded_feat.shape: ', encoded_feat.shape)

            # voxels_pred = self.decoder(encoded_feat) 

            # voxels_pred = self.layer1(encoded_feat)
            # voxels_pred = self.layer2(voxels_pred)
            # voxels_pred = self.layer3(voxels_pred)
            # voxels_pred = self.layer4(voxels_pred)
            
            voxels_pred = self.decoder(encoded_feat)
            voxels_pred = torch.reshape(voxels_pred, (args.batch_size, 1, 32, 32, 32))

            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)
            print('pointclouds_pred.shape: ', pointclouds_pred.shape)

            pointclouds_pred = torch.reshape(pointclouds_pred, (args.batch_size, args.n_points, 3))
            print('pointclouds_pred.shape: ', pointclouds_pred.shape)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          
