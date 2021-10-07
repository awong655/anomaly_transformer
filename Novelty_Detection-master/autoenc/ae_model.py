#from vit_pytorch import ViT
from TransGan_pytorch import models as TGModels
import fb_vit.vision_transformer as fb_vit
import torch.nn as nn

class autoencoder:
    # TODO: Set up configurations
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def get_encoder(self, encoder):
        return encoder
    def get_decoder(self, decoder):
        return decoder

class tmpViT:
    def __init__(self, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, dropout, emb_dropout, keep_head):

        self.ViT = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            keep_head=keep_head
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.keep_head = keep_head

    def get_ViT(self):
        return self.ViT
class facebook_vit():
    def __init__(self, image_size, patch_size, in_chans=3, num_classes=10, embed_dim=512
                 , depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm
                 ,keep_head=True):

        self.ViT = fb_vit.VisionTransformer(
            image_size=[image_size],
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            keep_head=keep_head
        )

    def get_ViT(self):
        return self.ViT

class tmpTransGan:
    def __init__(self, depth1, depth2, depth3, initial_size, dim, heads, mlp_ratio, drop_rate):
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.initial_size = initial_size
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.TransGan = TGModels.Generator(depth1=depth1, depth2=depth2, depth3=depth3, initial_size=initial_size,
                                           dim=dim, heads=heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
    def get_TransGan(self):
        return self.TransGan

