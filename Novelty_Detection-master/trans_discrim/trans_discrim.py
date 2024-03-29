import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) using the sigmoid activation function"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, kv=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        if kv is None:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        else:
            # passed key and value in as param
            kv = self.to_qkv(kv)
            kv = kv.chunk(3, dim = -1)
            _, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
            q, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, enc_attn=None):
        for attn, ff in self.layers:
            if enc_attn is None:
                x = attn(x) + x
            else:
                x = attn(x=x, kv=enc_attn) + x
            x = ff(x) + x
        return x

class ViT_Discrim(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, num_classes=1, pool = 'cls', channels = 3, dim_head = 64, dim_disc_head=32, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.patch_anom_mlp = MLP(input_dim=dim, hidden_dim=dim_disc_head, output_dim=1, num_layers=2)
        #self.anom_ff = MLP(input_dim=int(image_size/patch_size)**2, hidden_dim=dim_disc_head, output_dim=1, num_layers=1)
        self.anom_lin = nn.Linear(int(image_size/patch_size)**2, 1)

    def forward(self, img, enc_attn):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.transformer(x, enc_attn)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)

        # discard cls
        x = x[:,1:,:]

        # flatten patch embeddings into 2D tensor and feed through mlp
        # all patches in batch of images make up the rows
        # each patch fed through MLP to get prediction
        fdim, sdim, tdim = x.shape
        flt = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        x = self.patch_anom_mlp(flt) # patch wise anomaly prediction
        # reshape list of patch predictions to original number of images and patches
        # this groups all patch predictions into 1 row per image
        x = x.reshape((fdim, sdim)) # This is the patch prediction.
        patch_pred = x
        x = x.sum(dim=1, keepdim=True).squeeze()

        # Backprop on overall binary anomaly decision
        # Individual patch anomaly decision contribute to the overall binary anomaly decision
        # Therefore, patch anomaly segmentation is done in an unsupervised manner
        return x, patch_pred # return overall anomaly score

