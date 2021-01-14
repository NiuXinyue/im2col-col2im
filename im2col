import torch
import torch.nn.functional as F
def myUnFold(mtx, block_size, stride):
    n,c,h,w = mtx.shape
    block_size = block_size
    sx = (h - block_size[0]) //stride[0] + 1  #x轴最大索引
    sy = (w - block_size[1]) //stride[1] + 1  #y轴最大索引
    result = torch.zeros((n, c, block_size[0] * block_size[1], sx * sy))  #sx * sy是总共包含的块数
    for cc in range(c):
        img = mtx[:, cc]
        for j in range(sy):
            for i in range(sx):
                result[:, cc, :, j*sx + i] = img[:,i*stride[0]:(i*stride[0] + block_size[0]), j*stride[1]:(j*stride[1] + block_size[1])].reshape(-1, block_size[0] * block_size[1])
    return result.reshape(n*c, block_size[0] * block_size[1] , -1)
