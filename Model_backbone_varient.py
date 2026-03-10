import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
# -----------------------------
# LTC (your original, kept intact)
# -----------------------------
class LTC(nn.Module):
    def __init__(self, input_size, output_size, tau_min=0.1, tau_max=1.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_0 = tau_max  # Base time constant

        self.input_to_neurons = nn.Linear(input_size, output_size)

        self.W = nn.Parameter(torch.randn(output_size))
        self.gamma_I = nn.Parameter(torch.randn(output_size))
        self.gamma_r = nn.Parameter(torch.randn(output_size))
        self.mu = nn.Parameter(torch.randn(output_size))
        self.A = nn.Parameter(torch.randn(output_size))

        self.tau_input = nn.Linear(input_size + output_size, output_size)  # W_tau
        self.tau_bias = nn.Parameter(torch.randn(output_size) * 0.01)

    def activation(self, x_t, input_t):
        return torch.tanh(self.gamma_r * x_t + self.gamma_I * input_t + self.mu)

    def fused_step(self, u_t, x_t, delta_t):
        # dynamic tau
        concat_input = torch.cat([u_t, x_t], dim=-1)
        tau_raw = self.tau_input(concat_input) + self.tau_bias
        tau = self.tau_0 / (1 + torch.sigmoid(tau_raw))
        tau = torch.clamp(tau, self.tau_min, self.tau_0)

        f_x = self.activation(x_t, u_t)
        x_next = x_t + delta_t * f_x * self.A / (1 + delta_t * (1 / tau + f_x))
        return x_next

    def forward(self, x, L=100, return_hidden_states=False):
        """
        x: [B, D]; returns [B, D]
        We unfold L steps with constant input u_t = x, evolving hidden x_t.
        """
        delta_t = 1.0 / L
        x_t = self.input_to_neurons(x)
        hidden_states = [x_t.detach().cpu().numpy()] if return_hidden_states else None

        for _ in range(L):
            x_t = self.fused_step(x, x_t, delta_t)
            if return_hidden_states:
                hidden_states.append(x_t.detach().cpu().numpy())

        if return_hidden_states:
            return x_t, hidden_states
        return x_t

# -----------------------------
# Temporal selector (single interface)
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, kind: str, dim: int, steps: int = 100):
        super().__init__()
        self.kind = kind.upper()
        self.steps = steps

        if self.kind == "LTC":
            self.block = LTC(dim, dim)
        else:
            raise ValueError(f"Unknown temporal kind: {kind}")

    def forward(self, x: torch.Tensor):
        # Each block accepts [B,D] and internally uses self.steps
        if isinstance(self.block, LTC):
            return self.block(x, L=self.steps)
        else:
            raise RuntimeError("Unsupported block")

# ------------------------------------------------------------------


# -----------------------------
# DC/AC Conv Block (unchanged)
# -----------------------------
class DCACConvBlock(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()
        self.depthwise_conv = nn.Conv3d(
            input_channel,
            input_channel,
            kernel_size=5,
            padding=2,
            groups=input_channel
        )
        self.bn = nn.BatchNorm3d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        init.kaiming_normal_(self.depthwise_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.depthwise_conv.bias is not None:
            init.constant_(self.depthwise_conv.bias, 0)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ================================================================
# Backbones: all must implement forward(x[B,C,T,H,W]) -> feat[B,F]
# ================================================================

# -------- 2D ResNet-18 (framewise, then temporal pooling) --------
class ResNet2D18Backbone(nn.Module):
    def __init__(self, in_ch=3, out_dim=512, temporal_pool="mean"):
        super().__init__()
        # load backbone
        try:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            self.backbone = resnet18(weights=None)

        if in_ch != 3:
            # replace first conv to accept arbitrary channels
            conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_ch, conv1.out_channels,
                                            kernel_size=conv1.kernel_size,
                                            stride=conv1.stride,
                                            padding=conv1.padding,
                                            bias=False)
        # strip fc, keep global avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # -> [B,512,1,1]
        self.out_dim = out_dim
        self.temporal_pool = temporal_pool

    def forward(self, x):  # x: [B,C,T,H,W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()     # [B,T,C,H,W]
        x = x.view(B*T, C, H, W)                      # [B*T,C,H,W]
        feat = self.backbone(x).flatten(1)            # [B*T, 512]

        feat = feat.view(B, T, -1)                    # [B,T,512]
        if self.temporal_pool == "mean":
            feat = feat.mean(dim=1)                   # [B,512]
        elif self.temporal_pool == "max":
            feat = feat.max(dim=1).values
        else:
            # last-frame as fallback
            feat = feat[:, -1, :]
        return feat                                    # [B,512]


# -------- 3D ResNet-18 (r3d_18, spatiotemporal) --------
class ResNet3D18Backbone(nn.Module):
    """
    torchvision.models.video.r3d_18 backbone.
    Input : [B, C, T, H, W]
    Output: [B, 512]
    """
    def __init__(self, in_ch=3, out_dim=512, pretrained: bool = True):
        super().__init__()
        try:
            weights = R3D_18_Weights.DEFAULT if pretrained else None
            backbone = r3d_18(weights=weights)
        except Exception:
            backbone = r3d_18(weights=None)

        # If you need arbitrary channels, you may replace stem conv:
        # r3d_18 stem conv is in backbone.stem[0]
        if in_ch != 3:
            stem_conv = backbone.stem[0]
            backbone.stem[0] = nn.Conv3d(
                in_ch,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=False
            )

        # Remove classifier (avgpool+fc) -> keep up to last conv stage
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # [B,512,T',H',W']
        self.out_dim = out_dim

    def forward(self, x):  # [B,C,T,H,W]
        feat = self.backbone(x)                                   # [B,512,T',H',W']
        feat = F.adaptive_avg_pool3d(feat, (1,1,1)).flatten(1)     # [B,512]
        return feat



from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# # helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
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

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(
        self, *, image_size=32, image_patch_size, frames, frame_patch_size,
        dim, depth, heads, mlp_dim, channels=3, dim_head=64,
        dropout=0., emb_dropout=0.
    ):
        super().__init__()

        # store patch sizes for use in forward
        ih, iw = pair(image_size)
        ph, pw = pair(image_patch_size)
        self.patch_h, self.patch_w = ph, pw
        self.frame_patch = frame_patch_size

        assert ih % ph == 0 and iw % pw == 0, 'Image dims must be divisible by patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size.'

        # number of tokens based on *init* geometry (positional table will be sliced in forward)
        num_patches = (ih // ph) * (iw // pw) * (frames // frame_patch_size)
        patch_dim = channels * ph * pw * frame_patch_size

        # patch embed
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)',
                      p1=ph, p2=pw, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # learnable position embeddings (no CLS token now)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # per-frame head: map each frame token -> scalar
        self.per_frame_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, video):  # video: (b, c, f, h, w)
        b, _, F, H, W = video.shape
        ph, pw, pf = self.patch_h, self.patch_w, self.frame_patch

        assert (H % ph == 0) and (W % pw == 0), 'Input H/W must be divisible by patch sizes.'
        assert (F % pf == 0), 'Frames must be divisible by frame_patch_size.'

        # tokens
        x = self.to_patch_embedding(video)                 # (b, N, dim), N = (F/pf)*(H/ph)*(W/pw)
        n = x.shape[1]

        # add/slice positional embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # transformer over all tokens
        x = self.transformer(x)                            # (b, N, dim)

        # reshape to (b, F/pf, S, dim) with S = spatial tokens per frame
        Ht, Wt = H // ph, W // pw
        Ft = F // pf
        S = Ht * Wt
        x = x.view(b, Ft, S, -1)

        # spatial mean -> one token per (temporal) patch
        x = x.mean(dim=2)                                  # (b, Ft, dim)

        # if you want *exactly one* output per original frame, set pf=1 in init
        # head per frame
        y = self.per_frame_head(x).squeeze(-1)             # (b, Ft)

        return y

# --------- 2D ViT over frames ---------
class ViT2D_Video(nn.Module):
    """
    2D ViT applied per-frame.
    Input:  (B, C, T, H, W)
    Output: (B, T)  one scalar per frame
    """
    def __init__(
        self, *,
        image_size=128,               # reference H/W for max positional table; can differ at runtime if divisible by patch size
        image_patch_size=16,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        channels=3,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()

        ih, iw = pair(image_size)
        ph, pw = pair(image_patch_size)
        assert ih % ph == 0 and iw % pw == 0, "image_size must be divisible by patch size"

        self.patch_h, self.patch_w = ph, pw

        # tokens per frame at reference size (used to size pos embedding; sliced at runtime)
        tokens_per_frame_ref = (ih // ph) * (iw // pw)
        patch_dim = channels * ph * pw

        # patch embedding (2D only)
        self.to_patch_embedding = nn.Sequential(
            # (B, C, T, H, W) -> (B*T, H/Ph * W/Pw, Ph*Pw*C)
            Rearrange('b c t (h p1) (w p2) -> (b t) (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # learnable spatial positional embeddings (per-frame)
        self.pos_embedding = nn.Parameter(torch.randn(1, tokens_per_frame_ref, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # transformer over spatial tokens (per frame)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # head: one scalar per frame (use mean pooled token)
        self.per_frame_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, video):  # (B, C, T, H, W)
        b, c, t, h, w = video.shape
        ph, pw = self.patch_h, self.patch_w

        # sanity: divisible
        assert (h % ph == 0) and (w % pw == 0), "H/W must be divisible by patch size"

        # (B*T, S, D)
        x = self.to_patch_embedding(video)
        s = x.size(1)  # tokens per frame at runtime

        # add/slice spatial pos emb
        if self.pos_embedding.size(1) < s:
            # if runtime S exceeds reference, expand pos table (rare). Simple nearest repeat.
            repeat_factor = (s + self.pos_embedding.size(1) - 1) // self.pos_embedding.size(1)
            pos = self.pos_embedding.repeat(1, repeat_factor, 1)[:, :s]
        else:
            pos = self.pos_embedding[:, :s]

        x = x + pos
        x = self.emb_dropout(x)

        # transformer per frame (but we've folded frames into batch)
        x = self.transformer(x)                 # (B*T, S, D)

        # mean pool tokens -> (B*T, D)
        x = x.mean(dim=1)

        # map to scalar -> (B*T, 1) -> (B, T)
        y = self.per_frame_head(x).view(b, t)

        return y
    

# ---------------------- Tiny 2D ViT (framewise) -------------------
class ViT2DTinyBackbone(nn.Module):
    """
    Very light ViT:
      - Patch embed with Conv2d(patch=16)
      - 2 Transformer layers, 3 heads
      - CLS token per frame
      - Temporal mean over frames
    """
    def __init__(self, in_ch=3, embed_dim=192, patch=16, depth=2, heads=3, mlp_ratio=2.0):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = None  # initialized lazily once we see H,W
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_dim = embed_dim

    def _build_pos(self, H_p, W_p, device):
        N = H_p * W_p + 1
        self.pos = nn.Parameter(torch.zeros(1, N, self.out_dim, device=device))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):  # [B,C,T,H,W]
        B, C, T, H, W = x.shape
        x = x.permute(0,2,1,3,4).contiguous()      # [B,T,C,H,W]
        x = x.view(B*T, C, H, W)

        x = self.embed(x)                           # [B*T, D, H', W']
        _, D, Hp, Wp = x.shape
        if self.pos is None or self.pos.shape[1] != (Hp*Wp + 1):
            self._build_pos(Hp, Wp, x.device)

        tokens = x.flatten(2).transpose(1, 2)       # [B*T, N, D]
        cls = self.cls_token.expand(B*T, -1, -1)    # [B*T, 1, D]
        tokens = torch.cat([cls, tokens], dim=1) + self.pos  # add pos

        y = self.encoder(tokens)                    # [B*T, N, D]
        cls_out = y[:, 0, :]                        # [B*T, D]
        cls_out = cls_out.view(B, T, D).mean(dim=1) # temporal mean
        return cls_out                              # [B, D]


# ---------------------- Tiny 3D ViT (tubelets) --------------------
class ViT3DTinyBackbone(nn.Module):
    """
    Light 3D ViT:
      - Tubelet embed (t=4, p=16)
      - 2 Transformer layers, 3 heads
      - CLS token over space-time tokens
    """
    def __init__(self, in_ch=3, embed_dim=192, t_patch=4, p=16, depth=2, heads=3, mlp_ratio=2.0):
        super().__init__()
        self.t_patch = t_patch
        self.p = p
        self.embed3d = nn.Conv3d(in_ch, embed_dim, kernel_size=(t_patch, p, p),
                                 stride=(t_patch, p, p))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = None
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=int(embed_dim*mlp_ratio),
            batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.out_dim = embed_dim

    def _build_pos(self, Tt, Hp, Wp, device):
        N = Tt * Hp * Wp + 1
        self.pos = nn.Parameter(torch.zeros(1, N, self.out_dim, device=device))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):  # [B,C,T,H,W]
        z = self.embed3d(x)                         # [B,D,T',H',W']
        B, D, Tt, Hp, Wp = z.shape
        if self.pos is None or self.pos.shape[1] != (Tt*Hp*Wp + 1):
            self._build_pos(Tt, Hp, Wp, z.device)

        tokens = z.flatten(2).transpose(1, 2)       # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)      # [B,1,D]
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        y = self.encoder(tokens)                    # [B,N,D]
        cls_out = y[:, 0, :]                        # [B,D]
        return cls_out





# --------------------- Backbone selector/factory -------------------
class BackboneBlock(nn.Module):
    """
    kind: "RESNET2D18" | "RESNET3D18" | "VIT2D_TINY" | "VIT3D_TINY" | "PHYSNET"
    Returns feature vector [B, F] with .out_dim attribute.
    """
    def __init__(self, kind: str, in_ch: int = 3):
        super().__init__()
        kind = kind.upper()
        if kind == "RESNET3D18":
            self.block = ResNet3D18Backbone(in_ch=in_ch, out_dim=512, pretrained=True)
            self.out_dim = 512
        else:
            raise ValueError(f"Unknown backbone kind: {kind}")

    def forward(self, x):
        return self.block(x)


# ===========================
# Main model with pluggables
# ===========================
class SpO2Model(nn.Module):
    def __init__(self,
                 input_channel=3,
                 output_dim=300,
                 temporal_kind: str = "LTC",
                 temporal_steps: int = 100,
                 backbone_kind: str = "RESNET3D18"):
        super().__init__()

        # DC/AC front
        self.dc_conv = DCACConvBlock(input_channel=input_channel)
        self.ac_conv = DCACConvBlock(input_channel=input_channel)

        # Backbone (shared spec, instantiated twice for DC/AC)
        self.dc_backbone = BackboneBlock(kind=backbone_kind, in_ch=input_channel)
        self.ac_backbone = BackboneBlock(kind=backbone_kind, in_ch=input_channel)
        feat_dim = self.dc_backbone.out_dim

        # Fusion -> projection to output_dim
        self.linear = nn.Linear(2 * feat_dim, output_dim)

        # Temporal module (from your previous ablation code)
        self.temporal = TemporalBlock(kind=temporal_kind, dim=output_dim, steps=temporal_steps)

    def forward(self, x):
        # x: [B,C,T,H,W]
        x_dc = self.dc_conv(x)
        x_ac = self.ac_conv(x)

        f_dc = self.dc_backbone(x_dc)  # [B,F]
        f_ac = self.ac_backbone(x_ac)  # [B,F]

        fused = torch.cat([f_dc, f_ac], dim=1)  # [B, 2F]
        x_fused = self.linear(fused)            # [B, D=300]
        out = self.temporal(x_fused)            # [B, 300]
        return out, x_dc, x_ac


# -----------------------------
# Quick smoke test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, T, H, W = 1, 3, 300, 32, 32
    x = torch.randn(B, C, T, H, W)
    for bk in ["RESNET3D18"]:
        for kind in ["LTC"]:
            print(f"\n== Backbone: {bk}",f", Temporal kind: {kind} ==")
            model = SpO2Model(input_channel=3, output_dim=300,temporal_kind=kind,temporal_steps=100,
                            backbone_kind=bk)
            y, _, _ = model(x)
            print("Output:", y.shape)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(f"Trainable params: {params:.2f}M")


