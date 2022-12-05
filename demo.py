import torch # pytorch backend
import torchvision # CV models
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image
from model import GMNet
pygm.BACKEND = 'pytorch' # set default backend for pygmtools

def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = torch.zeros(len(kpt[0]), len(kpt[0]))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A


if __name__ == "__main__":
    obj_resize = (256, 256)
    img1 = Image.open('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0001.png')
    img2 = Image.open('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0002.png')
    kpts1 = torch.tensor(sio.loadmat('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0001.mat')['pts_coord'])
    kpts2 = torch.tensor(sio.loadmat('./data/WillowObject/WILLOW-ObjectClass/Duck/060_0002.mat')['pts_coord'])
    kpts1[0] = kpts1[0] * obj_resize[0] / img1.size[0]
    kpts1[1] = kpts1[1] * obj_resize[1] / img1.size[1]
    kpts2[0] = kpts2[0] * obj_resize[0] / img2.size[0]
    kpts2[1] = kpts2[1] * obj_resize[1] / img2.size[1]
    img1 = img1.resize(obj_resize, resample=Image.BILINEAR)
    img2 = img2.resize(obj_resize, resample=Image.BILINEAR)
    torch_img1 = torch.from_numpy(np.array(img1, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(0) # shape: BxCxHxW
    torch_img2 = torch.from_numpy(np.array(img2, dtype=np.float32) / 256).permute(2, 0, 1).unsqueeze(0) # shape: BxCxHxW
    A1 = delaunay_triangulation(kpts1)
    A2 = delaunay_triangulation(kpts2)
    
    model = GMNet()
    X = model(torch_img1, torch_img2, kpts1, kpts2, A1, A2)
    print(X)
    
    