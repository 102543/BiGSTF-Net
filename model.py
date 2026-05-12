import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math
from config import model_config
import copy
import time

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x



class PreBlock(torch.nn.Module):
    """
    Preprocessing module. It is designed to replace filtering and baseline correction.

    Args:
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
    """
    def __init__(self, sampling_point):
        super().__init__()
        self.pool1 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool2 = torch.nn.AvgPool1d(kernel_size=13, stride=1, padding=6)
        self.pool3 = torch.nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.ln_0 = torch.nn.LayerNorm(sampling_point)
        self.ln_1 = torch.nn.LayerNorm(sampling_point)

    def forward(self, x):
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]

        x0 = x0.squeeze()
        x0 = self.pool1(x0)
        x0 = self.pool2(x0)
        x0 = self.pool3(x0)
        x0 = self.ln_0(x0)
        x0 = x0.unsqueeze(dim=1)

        x1 = x1.squeeze()
        x1 = self.pool1(x1)
        x1 = self.pool2(x1)
        x1 = self.pool3(x1)
        x1 = self.ln_1(x1)
        x1 = x1.unsqueeze(dim=1)

        x = torch.cat((x0, x1), 1)

        return x


class fNIRS_T(nn.Module):
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = 100
        num_channels = 100
        self.temporal_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 30), stride=(1, 4)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Rearrange('b c h w  -> b h (c w)'),
            # output width * out channels --> dim
            nn.Linear((math.floor((sampling_point-30)/4)+1)*32, dim),
            nn.LayerNorm(dim))

        self.channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 30), stride=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Rearrange('b c h w  -> b h (c w)'),
            nn.Linear((math.floor((sampling_point-30)/4)+1)*32, dim),
            nn.LayerNorm(dim))
     
            
        self.pos_embedding_patch = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pos_embedding_channel = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        
        # def print_shape(module, input, output):
        #     print(f"Module: {module.__class__.__name__}, Input shape: {input[0].shape}, Output shape: {output.shape}")

        # for layer in self.temporal_embedding:
        #     layer.register_forward_hook(print_shape)


    def forward(self, img, mask=None):
        x = self.temporal_embedding(img)
        x2 = self.channel_embedding(img.squeeze())
        if not model_config.no_transformer:
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token_patch, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding_patch[:, :(n + 1)]
            x = self.dropout_patch(x)
            x = self.transformer_patch(x, mask)

            b, n, _ = x2.shape

            cls_tokens = repeat(self.cls_token_channel, '() n d -> b n d', b=b)
            x2 = torch.cat((cls_tokens, x2), dim=1)
            x2 += self.pos_embedding_channel[:, :(n + 1)]
            x2 = self.dropout_channel(x2)
            x2 = self.transformer_channel(x2, mask)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]
        else:
            x = x.mean(axis = 1)
            x2 = x2.mean(axis = 1)
        return x, x2

class EEG_T(nn.Module):
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        print("------------ init eeg model")
        num_patches = 100
        num_channels = 100
        channel_dim = 30
        self.temporal_embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 30), stride=(1, 4)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Rearrange('b c h w  -> b w (c h)'),
            # output width * out channels --> dim
            nn.LazyLinear(dim),
            nn.LayerNorm(dim))
        
        # def print_shape(module, input, output):
        #     print(f"Module: {module.__class__.__name__}, Input shape: {input[0].shape}, Output shape: {output.shape}")

        # for layer in self.temporal_embedding:
        #     layer.register_forward_hook(print_shape)

        self.channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Rearrange('b c h w  -> b h (c w)'),
            nn.LazyLinear(dim),
            nn.LayerNorm(dim))
                
        self.pos_embedding_patch = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pos_embedding_channel = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, n_class))


    def forward(self, img, mask=None):
        x = self.temporal_embedding(img)
        # print("img size: ", x.size())
        x2 = self.channel_embedding(img)
        if not model_config.no_transformer:
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token_patch, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding_patch[:, :(n + 1)]
            x = self.dropout_patch(x)
            x = self.transformer_patch(x, mask)

            b, n, _ = x2.shape

            cls_tokens = repeat(self.cls_token_channel, '() n d -> b n d', b=b)
            x2 = torch.cat((cls_tokens, x2), dim=1)
            x2 += self.pos_embedding_channel[:, :(n + 1)]
            x2 = self.dropout_channel(x2)
            x2 = self.transformer_channel(x2, mask)

            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]
        else:
            x = x.mean(axis = 1)
            x2 = x2.mean(axis = 1)
        # x = self.to_latent(x)
        # x2 = self.to_latent(x2)
        # x3 = torch.cat((x, x2), 1)
        return x, x2


class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_kv_eeg = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q_fnirs = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, eeg, fnirs):
        b, n, _, h = *fnirs.shape, self.heads
        q = self.to_q_fnirs(fnirs)
        k, v = self.to_kv_eeg(eeg).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 多模态交叉注意力融合模块
class MultiLevelFusion(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.cross_attn1 = CrossModalAttention(dim, heads, dim//(heads), dropout)
    def forward(self, eeg, fnirs):
        # 第一阶段融合
        eeg_to_fnirs = self.cross_attn1(eeg, fnirs)
        fnirs_to_eeg = self.cross_attn1(fnirs, eeg)
        
        eeg = eeg + eeg_to_fnirs
        fnirs = fnirs + fnirs_to_eeg

        return eeg, fnirs

## 动态特征融合模块
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()  # 输出0~1的权重
        )
    
    def forward(self, x1, x2):
        gate_weight = self.gate(torch.cat([x1, x2], dim=1))
        return gate_weight * x1 + (1 - gate_weight) * x2

class CrossGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()  # 输出0~1的权重
        )
    
    def forward(self, x1, x2):
        gate_weight = self.gate(torch.cat([x1, x2], dim=1))
        return gate_weight * x1 + (1 - gate_weight) * x2, gate_weight * x2 + (1 - gate_weight) * x1

class FNIRS_EEG_T(nn.Module):
    def __init__(self, n_class, sampling_point_eeg, sampling_point_fnirs, dim, depth, \
            heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        
        self.eeg_sub_model = EEG_T(n_class, sampling_point_eeg, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout)
        
        self.fnirs_sub_model = fNIRS_T(n_class, sampling_point_fnirs, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout)

        # 多级融合模块
        self.cross_modal_fusion = MultiLevelFusion(dim, depth, heads, mlp_dim, dropout)

        self.gate_fusion = GatedFusion(dim)
        self.mlp_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim * 2, n_class)
        )
        
        # 用于消融实验
        self.simple_cross_temporal = CrossGatedFusion(dim)
        self.simple_cross_channel = CrossGatedFusion(dim)
    
        self.apply(init_weights)
    
    def forward(self, eeg, fnirs, mask=None):
        
        eeg_features = self.eeg_sub_model(eeg, mask)
        
        fnirs_features = self.fnirs_sub_model(fnirs, mask)
        
        # print("eeg: ", eeg_features.size(), "fnirs: ", fnirs_features.size())
        # 调整维度用于融合
        eeg_feat1, eeg_feat2 = eeg_features
        fnirs_feat1, fnirs_feat2 = fnirs_features
        
        eeg_feat1 = eeg_feat1.unsqueeze(1)  # 添加时间维度
        eeg_feat2 = eeg_feat2.unsqueeze(1)  # 添加时间维度
        fnirs_feat1 = fnirs_feat1.unsqueeze(1)  # 添加时间维度
        fnirs_feat2 = fnirs_feat2.unsqueeze(1) # 添加时间维度
        # print("eeg size: ", eeg_feat1.size(), eeg_feat2.size())
        # print("fnirs size: ", fnirs_feat2.size(), fnirs_feat2.size())
        
        if model_config.no_cross_modal_fusion:
            eeg_1, fnirs_1 = eeg_feat1 + fnirs_feat1, eeg_feat1 + fnirs_feat1
            eeg_2, fnirs_2 = eeg_feat2 + fnirs_feat2, eeg_feat2 + fnirs_feat2
        else:
            # 多级融合
            eeg_1, fnirs_1 = self.cross_modal_fusion(eeg_feat1, fnirs_feat1)
            eeg_2, fnirs_2 = self.cross_modal_fusion(eeg_feat2, fnirs_feat2)
        
        if model_config.no_gate_fusion:
            fused1 = (eeg_1.squeeze() + eeg_2.squeeze()) / 2
            fused2 = (fnirs_1.squeeze() + fnirs_2.squeeze()) / 2
        else:
            fused1 = self.gate_fusion(eeg_1.squeeze(), eeg_2.squeeze())
            fused2 = self.gate_fusion(fnirs_1.squeeze(), fnirs_2.squeeze())
        
        fused3 = torch.cat((fused1, fused2), 1)
        res = self.mlp_head(fused3), fused3
        
        # print("fused1: ", fused1.size(), "fused2: ", fused2.size())
                
        # print(fused3.size())
        return res 
    def get_feature(self, eeg, fnirs, mask=None):
        eeg_features = self.eeg_sub_model(eeg, mask)
        fnirs_features = self.fnirs_sub_model(fnirs, mask)
        # print("eeg: ", eeg_features.size(), "fnirs: ", fnirs_features.size())
        # 调整维度用于融合
        eeg_feat1, eeg_feat2 = eeg_features
        fnirs_feat1, fnirs_feat2 = fnirs_features
        
        return eeg_feat1, eeg_feat2, fnirs_feat1, fnirs_feat2     
        
class EEG_BRANCH(nn.Module):
    def __init__(self, n_class, sampling_point_eeg, sampling_point_fnirs, dim, depth, \
            heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        
        self.eeg_sub_model = EEG_T(n_class, sampling_point_eeg, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout)
        
        self.mlp_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim * 2, n_class)
        )
        self.apply(init_weights)
    def forward(self, eeg, _, mask=None):
        eeg_features = self.eeg_sub_model(eeg, mask)
        # 调整维度用于融合
        eeg_feat1, eeg_feat2 = eeg_features
        eeg_feat = torch.cat([eeg_feat1, eeg_feat2], dim = 1) 
        # print(fused3.size())
        return self.mlp_head(eeg_feat), eeg_feat

class FNIRS_BRANCH(nn.Module):
    def __init__(self, n_class, sampling_point_eeg, sampling_point_fnirs, dim, depth, \
            heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        
        self.fnirs_sub_model = fNIRS_T(n_class, sampling_point_fnirs, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout)
        
        self.mlp_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim * 2, n_class)
        )
        self.apply(init_weights)
    def forward(self, _, fnirs, mask=None):
        fnirs_features = self.fnirs_sub_model(fnirs, mask)
        # 调整维度用于融合
        fnirs_feat1, fnirs_feat2 = fnirs_features
        fnirs_feat = torch.cat([fnirs_feat1, fnirs_feat2], dim = 1) 
        # print(fused3.size())
        return self.mlp_head(fnirs_feat), fnirs_feat
    
def init_weights(m):
    """递归初始化模型参数"""
    if isinstance(m, nn.LazyLinear):
        return
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)  # 小偏置防止Dead ReLU
            
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
           
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.in_proj_bias, 0.)
        nn.init.constant_(m.out_proj.bias, 0.)