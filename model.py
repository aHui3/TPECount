import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
import sys
import numpy as np
from models_vit import CrossAttentionBlock
from util.pos_embed import get_2d_sincos_pos_embed,positional_encoding
from util.transformer import TransformerEncoder
import open_clip
import copy
# from qwen import qwen2_vl
import math
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Normalize, Compose, Resize
from torchvision.transforms.functional import InterpolationMode
from visualize import save_heatmap

class TPECount(nn.Module):
    def __init__(
        self,
        encoder_output_tokens_num=196,
        cross_embed_dim=512,
        cross_depth=3,
        cross_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),            
    ):
        super().__init__()
        self.cross_input_tokens=encoder_output_tokens_num
        # print(self.cross_input_tokens)
        # print(cross_embed_dim)
        self.cross_pos_embed = nn.Parameter(
            torch.zeros(1, self.cross_input_tokens, cross_embed_dim), requires_grad=False)
        # self.ablation_nocross_linear=nn.Sequential(
        #     nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 512),
        #     nn.ReLU(inplace=True),
        # )
        # self.ablation_nocross_linear=nn.Linear(1,512)
        self.cross_block=nn.ModuleList(
            [
                CrossAttentionBlock(
                    cross_embed_dim,
                    cross_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(cross_depth)
            ]
        )
        self.cross_norm = norm_layer(cross_embed_dim)

        # self.self_cross_block=nn.ModuleList(
        #     [
        #         CrossAttentionBlock(
        #             cross_embed_dim,
        #             cross_num_heads,
        #             mlp_ratio,
        #             qkv_bias=True,
        #             qk_scale=None,
        #             norm_layer=norm_layer,
        #         )
        #         for _ in range(cross_depth)
        #     ]
        # )
        # self.self_cross_norm = norm_layer(cross_embed_dim)
        
        self.selfatt=TransformerEncoder(num_layers=6,emb_dim=cross_embed_dim,num_heads=8,layer_norm_eps=1e-5,norm=True)

        # self.affinemap_upsampler = nn.Sequential(
        #     nn.ConvTranspose2d(1, 4, kernel_size=5, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(4, 1, kernel_size=5, stride=2, padding=1),
        # )

        self.attn_weight=nn.Parameter(torch.ones(1, self.cross_input_tokens, 1), requires_grad=True)
        self.addn_biases=nn.Parameter(torch.zeros(1, self.cross_input_tokens, 1), requires_grad=True)

        self.decode_head0 = nn.Sequential(
            nn.Conv2d(cross_embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),#(512,512)
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),#(512,256)
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

        self.initialize_weights()
        self.clip_model = open_clip.create_model(
            "ViT-B-16", pretrained="/home/ljh/code/checkpoints/CLIP/CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin"
        )

        #Freeze all the weights of clip model
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False

        #Freeze all the weights of the text encoder.
        # vis_copy = copy.deepcopy(self.clip_model.visual)
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        # self.clip_model.visual = vis_copy

        # self.img_tokens_proj = nn.Linear(512, 64)

    def initialize_weights(self):
    # Initialize the positional embedding for the feature interaction module.
        cross_pos_embed = get_2d_sincos_pos_embed(
            self.cross_pos_embed.shape[-1],
            int(self.cross_input_tokens**0.5),
            cls_token=False,
        )
        # print("cross_pos_embed:",cross_pos_embed.shape)
        self.cross_pos_embed.data.copy_(
            torch.from_numpy(cross_pos_embed).float().unsqueeze(0)
        )

        # Initialize nn.Linear and nn.LayerNorm layers.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use Xavier uniform weight initialization following the official JAX ViT.
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    
    def forward_img_encoder(self, imgs):
        return self.clip_model.encode_image(imgs)
    
    def forward_text_encoder(self, texts):
        return self.clip_model.encode_text(texts)

    def get_similarity(self, fuse_tokens, text_tokens):
        similarity_map=F.cosine_similarity(fuse_tokens,text_tokens,dim=-1)#[B,196]
        similarity_map=similarity_map.unsqueeze(-1)#[B,196,1]
        return similarity_map

    def updata_similarity(self, similarity_map):
        B=similarity_map.size(0)
        updata_similarity_map = self.attn_weight.expand(B, -1, -1) * similarity_map+self.addn_biases.expand(B, -1, -1)
        return updata_similarity_map

    def select_examplar(self,similarity_map,fuse_tokens):#[1,1,14,14],[1,196,512]
        _, topk_indices = torch.topk(similarity_map, k=4, dim=1, sorted=True)
        examplar_tokens = torch.gather(fuse_tokens, dim=1, index=topk_indices.expand(-1, -1, fuse_tokens.shape[-1]))
        final_examplar_token=torch.mean(examplar_tokens,dim=1).unsqueeze(1)
        return final_examplar_token


    def forward_cross(self, img_tokens, examplar_tokens,affine_attn_map):#
        # Add positional embedding to image tokens.
        img_tokens = img_tokens + self.cross_pos_embed
        x = img_tokens
        x=x+x*affine_attn_map#[1,196,512]
        for i,blk in enumerate(self.cross_block):
            x = blk(x, examplar_tokens)
            x=x+x*affine_attn_map
            
        return self.cross_norm(x)

    
    def forward_decoder(self,cross_output):
        # 获取cross_output的形状
        n, hw, c = cross_output.shape
        # 计算h和w的值
        h = w = int(math.sqrt(hw))
        # 将cross_output进行转置和reshape操作
        x = cross_output.transpose(1, 2).reshape(n, c, h, w)

        # Upsample output of this map to be N x [fim_embed_dim] x 24 x 24, as it was in CounTR.
        x = F.interpolate(x, size=24, mode="bilinear", align_corners=False)
        #F.interpolate() adjust the size of the input tensor to the given size.
        # Pass [x] through the density map regression decoder and upsample output until density map is the size of the input image.
        x = F.interpolate(
            self.decode_head0(x), 
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head1(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head2(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.interpolate(
            self.decode_head3(x),
            size=x.shape[-1] * 2,
            mode="bilinear",
            align_corners=False,
        )

        # Remove the channel dimension from [x], as the density map only has 1 channel.
        return x.squeeze(-3)
    
    def forward(self,clip_image_inputs,clip_text_inputs):#qwen_inputs, clip_image_inputs, clip_text_inputs
        img_tokens = self.forward_img_encoder(clip_image_inputs)#[1, 196, 512]

        text_tokens = self.forward_text_encoder(clip_text_inputs).unsqueeze(-2)

        
        combined_tokens = torch.cat([img_tokens, text_tokens], dim=1)
        pos=positional_encoding(combined_tokens)
        fuse_tokens=self.selfatt(combined_tokens,pos,src_mask=None, src_key_padding_mask=None)#[1,197,512]
        crop_fuse_tokens=fuse_tokens[:,0:-1,:]#[1, 196, 512]

        # pos=positional_encoding(img_tokens)
        # crop_fuse_tokens=self.forward_self_cross(img_tokens,text_tokens,pos)
        
        # crop_fuse_tokens=crop_fuse_tokens+img_tokens

        similarity_map=self.get_similarity(crop_fuse_tokens, text_tokens)#[B,196,1]，crop_fuse_tokens
        examplar=self.select_examplar(similarity_map,crop_fuse_tokens)#[1, 1, 512]，crop_fuse_tokens


        affine_attn_map=self.get_similarity(img_tokens,examplar)#examplar

        affine_attn_map=self.updata_similarity(affine_attn_map)

        cross_output=self.forward_cross(img_tokens,examplar,affine_attn_map)#img_tokens,examplar,affine_attn_map

        pre=self.forward_decoder(cross_output)#[1, 384, 384]supervised_cross_output
        return pre












    
