import torch
import torch.nn.functional as F
def myFold(mtx, image_size, block_size, stride):
    nc, h, w = mtx.shape
    image_size = image_size
    sx = (image_size[0] - block_size[0])//stride[0] + 1
    sy = (image_size[1] - block_size[1])//stride[1] + 1
    result = torch.zeros([nc, image_size[0], image_size[1]])#3 4 4
    for j in range(sy):
        for i in range(sx):
            result[:, i*stride[0]:(i*stride[0] + block_size[0]),j*stride[1]:(j*stride[1] + block_size[1])] += mtx[:, :, j*sx + i].reshape([-1, *block_size])
    return result
