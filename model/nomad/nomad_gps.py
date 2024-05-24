import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from model.nomad.base_module import PositionalEncoding, MLP 

class NoMaD_GPS(nn.Module):
    '''
        backbone encoder for NoMAD
    '''
    def __init__(
        self,
        context_size: int = 20,   # 20
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 256, # 256
        mha_num_attention_heads: Optional[int] = 4, # 4
        mha_num_attention_layers: Optional[int] = 4,    # 4
        mha_ff_dim_factor: Optional[int] = 4,   # 4
    ) -> None:
        """
        NoMaD Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        #-------------------------------------observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        # compression layers 
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        #------------------------------------the goal encoder

        # MLP : input_dim, hidden_dim, output_dim, num_hidden_layers
        self.goal_encoder = MLP(2, 4, 2, 2)
        self.img_to_goal_endcoder = nn.Linear(self.goal_encoding_size, 2)
        self.goal_pos_encoder = MLP(44, 128, 64, 2)
        
        # --------------------------------- positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        #----------------------------------- TODO: initalize weights
        self._init_weights()

    def _init_weights(self) -> None:
        pass

    def forward(self, 
                obs_img: torch.tensor, 
                goal_pos: torch.tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        # obs_img: [B, C*N, H, W], output: [B, D], D=context_size, default=256
        device = obs_img.device


        #----------------------------1. encode obs image (past 20 + cur 1)----------------------------#
        obs_img = torch.split(obs_img, 3, dim=1)    # [B, C*N, H, W] ---> list of [B, N, H, W]
        obs_img = torch.concat(obs_img, dim=0)      # [B*C, N, H, W]

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)                  # --> [B, 21, 256]

        #----------------------------2. encode goal image----------------------------#

        # goal_pos [B,2] in ego coord
        img_to_goal = self.img_to_goal_endcoder(obs_encoding)               # linear : [B, 21, 256] ----> [B, 21, 2]
        goal_pos = self.goal_encoder(goal_pos)                             # MLP : [B, 2] ---> [B, 2]
        goal_pos = torch.cat([goal_pos, img_to_goal.view(-1, 42) ], dim=1)  # [B, 21*2+2] = [B, 44]
        goal_pos = self.goal_pos_encoder(goal_pos)                          # MLP : [B, 44] ---> [B, 64]
        goal_pos = goal_pos.repeat(1, 4).unsqueeze(1)                       # [B, 64] ---> [B, 1, 128]


        #----------------------------. cat tokens ( query )------------------------------------
        obs_encoding = torch.cat((obs_encoding, goal_pos), dim=1)            # --> [B, 22, 256]
        
        # positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        #----------------------------3. transformer encoder ----------------------------#
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=None)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)         # --> [B, 256]

        return obs_encoding_tokens


# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module



    