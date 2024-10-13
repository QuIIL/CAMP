from torch import nn
from .r2t_emb_position import *
from .r2t_rmsa import *
from timm.models.layers import DropPath

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,attn='rmsa',n_region=8,epeg=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,crmsa_k=3,epeg_k=15,**kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        # if attn == 'ntrans':
        #     self.attn = NystromAttention(
        #         dim = dim,
        #         dim_head = trans_dim,  # dim // 8
        #         heads = head,
        #         num_landmarks = 256,    # number of landmarks dim // 2
        #         pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #         residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #         dropout=drop_out
        #     )
        if attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

    def forward(self,x,need_attn=False):

        x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x,attn

class RRTEncoder(nn.Module):
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='rmsa', \
                region_num=8,drop_out=0.1,n_layers=2,n_heads=8,drop_path=0., \
                ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,epeg=True, \
                epeg_k=15,region_size=0,min_region_num=0,min_region_ratio=0, \
                qkv_bias=True,peg_bias=True,peg_1d=False,cr_msa=True,crmsa_k=3, \
                all_shortcut=False,crmsa_mlp=False,crmsa_heads=8,need_init=False,**kwargs):
        super(RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = []
        for i in range(n_layers-1):
            self.layers += [TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=region_num,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,**kwargs)]
        self.layers = nn.Sequential(*self.layers)
    
        # CR-MSA
        self.cr_msa = TransLayer(dim=mlp_dim,head=crmsa_heads,drop_out=drop_out,
                                 drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,
                                 trans_dim=trans_dim,attn='crmsa',qkv_bias=qkv_bias,crmsa_k=crmsa_k,
                                 crmsa_mlp=crmsa_mlp,epeg_k=epeg_k,**kwargs) if cr_msa else nn.Identity()

        # only for ablation
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        # if need_init:
        #     self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4

        batch, num_patches, C = x.shape 
        x_shortcut = x

        # PEG/PPEG
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        
        # R-MSA within region
        for i,layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x+x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1,2)
            x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
        return x