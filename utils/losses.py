import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from typing import Tuple
import pytorch_ssim

def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        loss = -(self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean())
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, reduction: str = 'mean', max_val: float = 1.0) -> None:
        super(SSIMLoss, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2

    def filter2D(
            self,
            input: torch.Tensor,
            kernel: torch.Tensor,
            channel: int) -> torch.Tensor:
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss
# ------------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class L_grad(nn.Module):

    def __init__(self):
        super(L_grad, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self,x,y):
        D_org_right = F.conv2d(x , self.weight_right, padding="same")
        D_org_down = F.conv2d(x , self.weight_down, padding="same")
        D_enhance_right = F.conv2d(y , self.weight_right, padding="same")
        D_enhance_down = F.conv2d(y , self.weight_down, padding="same")
        return torch.abs(D_org_right),torch.abs(D_enhance_right),torch.abs(D_org_down),torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2


    def forward(self, x, y):

        x_R1,y_R1, x_R2,y_R2  = self.gradient_of_one_channel(x[:,0:1,:,:],y[:,0:1,:,:])
        x_G1,y_G1, x_G2,y_G2  = self.gradient_of_one_channel(x[:,1:2,:,:],y[:,1:2,:,:])
        x_B1,y_B1, x_B2,y_B2  = self.gradient_of_one_channel(x[:,2:3,:,:],y[:,2:3,:,:])
        x = torch.cat([x_R1,x_G1,x_B1,x_R2,x_G2,x_B2],1)
        y = torch.cat([y_R1,y_G1,y_B1,y_R2,y_G2,y_B2],1)
        loss = self.gradient_Consistency_loss_patch(x,y)
        return loss

class L_bright(nn.Module):

    def __init__(self):
        super(L_bright, self).__init__()

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        loss = self.gradient_Consistency_loss_patch(x,y)
        return loss


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x, y):
        
        product_separte_color = (x*y).mean(1,keepdim=True)
        x_abs = (x**2).mean(1,keepdim=True)**0.5
        y_abs = (y**2).mean(1,keepdim=True)**0.5
        loss = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        return torch.mean(loss)
    
class L_ssim(nn.Module):
    
    def __init__(self):
        super(L_ssim, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, high):
        loss = (1- self.ssim_loss(R_low,high)).mean()
        return loss

