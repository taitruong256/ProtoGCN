import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 view_num=11,
                 ratio=0.125,
                 intra_act='softmax',
                 inter_act='tanh',
                 use_euclid=True,
                 euclid_scale=1.0,
                 view_scale=1.0,
                 euclid_tau=1.0,
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.view_num = view_num
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)
        self.intra_act = intra_act
        self.inter_act = inter_act
        self.use_euclid = use_euclid
        self.euclid_scale = nn.Parameter(torch.tensor(float(euclid_scale)))
        self.view_scale = nn.Parameter(torch.tensor(float(view_scale)))
        self.euclid_tau = nn.Parameter(torch.tensor(float(euclid_tau)))

        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)
        self.view_conv = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        self.view_gap = nn.AdaptiveAvgPool2d(1)
        self.view_fc = nn.Linear(mid_channels * num_subsets, view_num)
        self.view_softmax = nn.Softmax(dim=-1)
        self.view_mats = nn.Parameter(A.clone().unsqueeze(0).repeat(view_num, 1, 1, 1))

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        
        n, c, t, v = x.shape
        res = self.down(x)
        # Predict view logits from the current feature map.
        view_feat = self.view_gap(self.view_conv(x)).view(n, -1)
        view_logits = self.view_fc(view_feat)
        view_prob = self.view_softmax(view_logits)
        self.last_view_logits = view_logits
        self.last_view_prob = view_prob

        # K V V
        A = self.A
        # 1 K 1 1 V V
        A = A[None, :, None, None] 
        # N K V V -> N K 1 1 V V
        A_view = torch.einsum('nv,vkxy->nkxy', view_prob, self.view_mats)
        A_view = A_view[:, :, None, None]
        A = A + self.view_scale * A_view

        euclid_graph = None
        if self.use_euclid:
            # Build a joint-to-joint distance matrix from the current features.
            joint_feat = x.mean(dim=2).transpose(1, 2).contiguous()  # N V C
            euclid_dist = torch.cdist(joint_feat, joint_feat, p=2)   # N V V
            tau = F.softplus(self.euclid_tau) + 1e-6
            euclid_graph = torch.exp(-euclid_dist / tau)
            euclid_graph = (self.euclid_scale * euclid_graph)[:, None, None, None, :, :]
            euclid_graph = euclid_graph.expand(-1, self.num_subsets, -1, -1, -1, -1)
            A = A + euclid_graph
        
        """
        ***********************************
        *** Motion Topology Enhancement ***
        ***********************************
        """
        # The shape of pre_x is N, K, C, T, V
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v) 
        x1, x2 = None, None
        # N C T V
        tmp_x = x
        # N K C T V
        x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        # N K C 1 V
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)
        graph_list = []
        # N K C 1 V V = N K C 1 V 1 - N K C 1 1 V
        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        # N K C 1 V V
        inter_graph = getattr(self, self.inter_act)(diff)
        inter_graph = inter_graph * self.alpha[0]
        # N K C 1 V V = N K C 1 V V + 1 K 1 1 V V
        A = inter_graph + A
        graph_list.append(inter_graph)
        # N K C 1 V * N K C 1 V = N K 1 1 V V
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        # N K 1 1 V V
        intra_graph = getattr(self, self.intra_act)(intra_graph)
        intra_graph = intra_graph * self.beta[0]
        # N K C 1 V V = N K 1 1 V V + N K C 1 V V
        A = intra_graph + A
        graph_list.append(intra_graph)
        A = A.squeeze(3)
        # N K C T V = N K C T V * N K C V V
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
        # N K C T V -> N K*C T V
        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        """
        ***********************************
        ***********************************
        ***********************************
        """
        
        get_gcl_graph = graph_list[0] + graph_list[1]
        get_gcl_graph = get_gcl_graph + self.view_scale * A_view
        if euclid_graph is not None:
            get_gcl_graph = get_gcl_graph + euclid_graph
        # N K C 1 V V -> N K C V V
        get_gcl_graph = get_gcl_graph.squeeze(3)
        # N K C V V -> N K*C V V
        get_gcl_graph = get_gcl_graph.reshape(n, -1, v, v)
        
        return self.act(self.bn(x) + res), get_gcl_graph
