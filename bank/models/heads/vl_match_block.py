from torch import nn, concat
from mmseg.models.builder import MODELS


@MODELS.register_module()
class VLMatch(nn.Module):
    def __init__(
        self,
        embed_dim,
    ):
        super(VLMatch, self).__init__()
        self.x_pre_linear = nn.Linear(768, embed_dim)
        self.lshadow_pre_linear = nn.Linear(512, embed_dim)
        self.lbg_pre_linear = nn.Linear(512, embed_dim)
        self.x2shadow = nn.MultiheadAttention(
            embed_dim, 8, dropout=0.1, batch_first=True
        )

        self.x2bg = nn.MultiheadAttention(embed_dim, 8, dropout=0.1, batch_first=True)
        self.bg2shadow = nn.MultiheadAttention(
            embed_dim, 8, dropout=0.1, batch_first=True
        )
        self.fuse_shadow = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.bg_post_linear = nn.Linear(embed_dim, embed_dim)
        # self.x_post_linear = nn.Linear(embed_dim, 768)
        # self.lshadow_post_linear = nn.Linear(embed_dim, 512)

    def forward(self, x, l_shadow, l_bg=None):
        # x: BT,N,768    l_shadow/l_bg: B,L,512
        # print(f"x:{x.shape},l_shadow:{l_shadow.shape},l_bg:{l_bg.shape}")
        t = 5
        b = x.shape[0] // t
        N = x.shape[1]
        x = x.reshape(b, t, N, -1).flatten(1, 2)  # B,TN,hidden_dims
        x = self.x_pre_linear(x)  # B,TN,embed_dim
        l_shadow = self.lshadow_pre_linear(l_shadow)  # B,L,embed_dim
        l_bg = self.lbg_pre_linear(l_bg)  # B,L,embed_dim


        # x = einops.rearrange(x, "n (b t) c -> (n t) b c", t=t)  # NT,B,embed_dim
        l_shadow = self.x2shadow(query=l_shadow, key=x, value=x)[0]

        l_bg = self.x2bg(query=l_bg, key=x, value=x)[0]
        # l_bg = self.norm2(l_bg)
        l_shadow_ = self.bg2shadow(query=l_shadow, key=l_bg, value=l_bg)[0]
        # l_shadow_ = self.norm3(l_shadow_) 
        l_shadow = self.fuse_shadow(
            concat((l_shadow, l_shadow_), dim=-1)
        )  # B,L,embed_dim
        l_bg = self.bg_post_linear(l_bg)  # B,L,embed_dim

        return l_shadow, l_bg
