import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import einops
from einops import rearrange
import numpy as np


# SplitPointMlp
class SplitPointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden_dim = int(dim//2 * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(dim//2, hidden_dim, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim//2, 1, 1, 0),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.fc(x1)
        x = torch.cat([x1, x2], dim=1)
        return x

# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type = 'WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

# Shuffle Mixing layer
class SMLayer(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.spatial = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.mlp1 = SplitPointMlp(dim, mlp_ratio)
        self.mlp2 = SplitPointMlp(dim, mlp_ratio)

    def forward(self, x):
        x = self.mlp1(self.norm1(x)) + x
        x = self.spatial(x)
        x = self.mlp2(self.norm2(x)) + x
        return x
    
# Feature Mixing Block
class FMBlock(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=2):
        super().__init__()
        self.net = nn.Sequential(
            SMLayer(dim, kernel_size, mlp_ratio),
            SMLayer(dim, kernel_size, mlp_ratio),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim + 16, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim + 16, dim, 1, 1, 0)
        )

    def forward(self, x):
        x = self.net(x) + x
        x = self.conv(x) + x
        return x

# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)
        return

    def forward(self, x):
        # x: [n, c, h, w]
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads
        reshape = lambda x: einops.rearrange(x, "n (nh dh) h w -> (n nh h) w dh", nh=n_heads, dh=dim_head)

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = map(reshape, qkv.chunk(3, dim=1)) # 在 1 维度上分为 3 块
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # fac = dim_head ** -0.5
        res = k.transpose(-2, -1)
        res = torch.matmul(q, res) * self.fac
        res = torch.softmax(res, dim=-1)

        res = torch.matmul(res, v)
        res = einops.rearrange(res, "(n nh h) w dh -> n (nh dh) h w", nh=n_heads, dh=dim_head, n=n, h=h)
        res = self.fin(res)

        return res

# Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=1, bias=True) -> None:
        super().__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)
        return

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4

        x = self.row_att(x)
        x = x.transpose(-2, -1)
        x = self.col_att(x)
        x = x.transpose(-2, -1)

        return x


# Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x


# Axis-based Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2)) # 这里先把通道数减半，然后再 PixelUnshuffle

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
# Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):  
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # @ 表示常规的数学上定义的矩阵相乘
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out
    
# ------------------- LLFormer ------------------- 
class LLFormer(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 16,
        num_blocks = [1,2,4,8],
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        attention=True,
        skip = False
    ):

        super(LLFormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = nn.Sequential(*[FMBlock(dim = dim, kernel_size = 7, mlp_ratio = 2) for _ in range(num_blocks[0])])
        self.encoder_2 = nn.Sequential(*[FMBlock(dim = dim, kernel_size = 7, mlp_ratio = 2) for _ in range(num_blocks[0])])
        self.encoder_3 = nn.Sequential(*[FMBlock(dim = dim, kernel_size = 7, mlp_ratio = 2) for _ in range(num_blocks[0])])

        # self.encoder_1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.encoder_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.encoder_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.layer_fussion = LAM_Module_v2(in_dim = dim * 3)
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.down_1 = Downsample(int(dim)) ## From Level 0 to Level 1
        self.decoder_level1_0 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.down_2 = Downsample(int(dim *2)) ## From Level 1 to Level 2
        self.decoder_level2_0 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.down_3 = Downsample(int(dim * 2*2)) ## From Level 2 to Level 3
        self.decoder_level3_0 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 4)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        # self.down_4 = Downsample(int(dim * 2 * 4)) ## From Level 3 to Level 4
        # self.decoder_level4 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 8)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])])

        # self.up4_3 = Upsample(int(dim * 2 *8))  ## From Level 4 to Level 3
        # self.decoder_level3_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 4)), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 *4)) ## From Level 3 to Level 2
        self.decoder_level2_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2 * 2)), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 *2)) ## From Level 2 to Level 1
        self.decoder_level1_1 = nn.Sequential(*[TransformerBlock(dim=int(int(dim * 2)), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.up2_0 = Upsample(int(dim * 2))  ## From Level 1 to Level 0

        ### skip connection wit weights
        # self.coefficient_4_3 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 4))))), requires_grad=attention)
        self.coefficient_3_2 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 2))))), requires_grad=attention)
        self.coefficient_2_1 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2))))), requires_grad=attention)
        self.coefficient_1_0 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim))))), requires_grad=attention)

        ### skip then conv 1x1
        # self.skip_4_3 = nn.Conv2d(int(int(dim * 2 * 4)), int(int(dim * 2 * 4)), kernel_size=1, bias=bias)
        self.skip_3_2 = nn.Conv2d(dim * 2 * 2, dim * 2 * 2, kernel_size=1, bias=bias)
        self.skip_2_1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.skip_1_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.refinement_1 = nn.Sequential(*[FMBlock(dim = dim, kernel_size = 7, mlp_ratio = 2) for _ in range(num_refinement_blocks)])
        self.refinement_2 = nn.Sequential(*[FMBlock(dim = dim, kernel_size = 7, mlp_ratio = 2) for _ in range(num_refinement_blocks)])
        self.refinement_3 = nn.Sequential(*[FMBlock(dim = dim, kernel_size = 7, mlp_ratio = 2) for _ in range(num_refinement_blocks)])
        # self.refinement_1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        # self.refinement_2 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        # self.refinement_3 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.layer_fussion_2 = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss_2 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.skip = skip


    def forward(self, inp_img):

        inp_enc_encoder1 = self.patch_embed(inp_img) # 3x3 conv output
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)


        inp_fusion_123 = torch.cat([out_enc_encoder1.unsqueeze(1),out_enc_encoder2.unsqueeze(1),out_enc_encoder3.unsqueeze(1)],dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)


        inp_enc_level1_0 = self.down_1(out_fusion_123)
        out_enc_level1_0 = self.decoder_level1_0(inp_enc_level1_0)



        inp_enc_level2_0 = self.down_2(out_enc_level1_0)
        out_enc_level2_0 = self.decoder_level2_0(inp_enc_level2_0)



        inp_enc_level3_0 = self.down_3(out_enc_level2_0)
        out_enc_level3_0 = self.decoder_level3_0(inp_enc_level3_0)



        # inp_enc_level4_0 =   self.down_4(out_enc_level3_0)
        # out_enc_level4_0 = self.decoder_level4(inp_enc_level4_0)


        # out_enc_level4_0 = self.up4_3(out_enc_level4_0)

        # inp_enc_level3_1 = self.coefficient_4_3[0, :][None, :, None, None] * out_enc_level3_0 + self.coefficient_4_3[1, :][None, :, None, None] * out_enc_level4_0
        # inp_enc_level3_1 = self.skip_4_3(inp_enc_level3_1)  ### conv 1x1


        # out_enc_level3_1 = self.decoder_level3_1(inp_enc_level3_1)


        # out_enc_level3_1 = self.up3_2(out_enc_level3_1)

        out_enc_level3_0 = self.up3_2(out_enc_level3_0)

        inp_enc_level2_1 = self.coefficient_3_2[0, :][None, :, None, None] * out_enc_level2_0 + self.coefficient_3_2[1, :][None, :, None, None] * out_enc_level3_0
        inp_enc_level2_1 = self.skip_3_2(inp_enc_level2_1)  ### conv 1x1


        out_enc_level2_1 = self.decoder_level2_1(inp_enc_level2_1)

        out_enc_level2_1 = self.up2_1(out_enc_level2_1)

        inp_enc_level1_1 = self.coefficient_2_1[0, :][None, :, None, None] * out_enc_level1_0 + self.coefficient_2_1[1, :][None, :, None, None] *  out_enc_level2_1

        inp_enc_level1_1 = self.skip_2_1(inp_enc_level1_1)  ### conv 1x1

        out_enc_level1_1 = self.decoder_level1_1(inp_enc_level1_1)

        out_enc_level1_1 = self.up2_0(out_enc_level1_1)


        # out_fusion_123 = self.latent(out_fusion_123)


        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123  + self.coefficient_1_0[1, :][None, :, None, None] *  out_enc_level1_1
        # out = self.skip_1_0(out)  ### conv 1x1

        out_1 = self.refinement_1(out)

        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)
        inp_fusion = torch.cat([out_1.unsqueeze(1),out_2.unsqueeze(1),out_3.unsqueeze(1)],dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)

        if self.skip:
            out = self.output(out)+ inp_img
        else:
            out = self.output(out)

        return out
    
if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # 720p [1280 * 720]
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    x = torch.randn(1, 3, 256, 256)

    model = LLFormer(inp_channels=3,out_channels=3,dim = 16,num_blocks = [2,4,8,16],num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',attention=True,skip = True)
    # print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(f'output: {output.shape}')

