import copy as cp
import logging
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.runner import load_checkpoint
from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import unit_gcn, mstcn, unit_tcn

EPS = 1e-4

logger = logging.getLogger(__name__)


def _shape(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (tuple, list)):
        return [_shape(i) for i in x]
    return type(x).__name__


class GCN_Block(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 reduction=4,
                 use_bottleneck=True,
                 **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        if 'view_num' in kwargs:
            kwargs['gcn_view_num'] = kwargs.pop('view_num')
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        if reduction < 1:
            raise ValueError(f'reduction must be >= 1, got {reduction}')

        self.use_bottleneck = use_bottleneck
        self.reduction = reduction
        self.bottleneck_channels = max(out_channels // reduction, 1)
        self.inner_channels = self.bottleneck_channels if self.use_bottleneck else out_channels
        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)

        self.conv_down = nn.Conv2d(in_channels, self.inner_channels, kernel_size=1)
        self.bn_down = build_norm_layer(norm_cfg, self.inner_channels)[1]
        self.conv_up = nn.Conv2d(self.inner_channels, out_channels, kernel_size=1) if self.use_bottleneck else nn.Identity()
        self.bn_up = build_norm_layer(norm_cfg, out_channels)[1]

        self.gcn = unit_gcn(self.inner_channels, self.inner_channels, A, **gcn_kwargs)
        self.tcn = mstcn(self.inner_channels, self.inner_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        if not residual:
            self.residual_gcn = lambda x: 0
            self.residual_tcn = lambda x: 0
        else:
            self.residual_gcn = lambda x: x
            if stride == 1:
                self.residual_tcn = lambda x: x
            else:
                self.residual_tcn = unit_tcn(
                    self.inner_channels,
                    self.inner_channels,
                    kernel_size=1,
                    stride=stride,
                )

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        logger.debug("GCN_Block.forward: in=%s", _shape(x))
        res = self.residual(x)
        x = self.relu(self.bn_down(self.conv_down(x)))
        res_gcn = self.residual_gcn(x)
        x, gcl_graph = self.gcn(x, A)
        logger.debug(
            "GCN_Block.forward: gcn_out=%s residual=%s res_gcn=%s",
            _shape(x),
            _shape(res),
            _shape(res_gcn),
        )
        tcn_in = x + res_gcn
        tcn_out = self.tcn(tcn_in)
        res_tcn = self.residual_tcn(tcn_in)
        logger.debug(
            "GCN_Block.forward: mstcn_out=%s res_tcn=%s",
            _shape(tcn_out),
            _shape(res_tcn),
        )
        x = self.relu(tcn_out + res_tcn)
        x = self.bn_up(self.conv_up(x))
        out = self.relu(x + res)
        logger.debug("GCN_Block.forward: out=%s graph=%s", _shape(out), _shape(gcl_graph))
        return out, gcl_graph


"""
****************************************
*** Prototype Reconstruction Network ***
****************************************
"""  
class Prototype_Reconstruction_Network(nn.Module):
    
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias = False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias = False)
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        logger.debug("Prototype_Reconstruction_Network.forward: in=%s", _shape(x))
        query = self.softmax(self.query_matrix(x), dim=-1)
        z = self.memory_matrix(query)
        out = self.dropout(z)
        logger.debug("Prototype_Reconstruction_Network.forward: out=%s", _shape(out))
        return out


@BACKBONES.register_module()
class ProtoGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 multi_branch=False,
                 multi_branch_stages=2,
                 branch_in_channels=3,
                 use_bottleneck=True,
                 block_variant='legacy',
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.multi_branch = multi_branch
        self.multi_branch_stages = multi_branch_stages
        self.branch_in_channels = branch_in_channels
        self.use_bottleneck = use_bottleneck
        self.block_variant = block_variant
        self.kwargs = kwargs
        self.num_person = num_person
        logger.info('ProtoGCN block_variant: %s | use_bottleneck=%s', self.block_variant, self.use_bottleneck)

        if self.multi_branch:
            if in_channels % branch_in_channels != 0:
                raise ValueError(
                    f'multi_branch=True expects in_channels to be divisible by '
                    f'branch_in_channels, got in_channels={in_channels} and '
                    f'branch_in_channels={branch_in_channels}.'
                )
            self.branch_num = in_channels // branch_in_channels
            self.branch_data_bn = nn.ModuleList([
                self._build_data_bn(num_person, branch_in_channels, A.size(1))
                for _ in range(self.branch_num)
            ])
        else:
            self.branch_num = 1
            self.data_bn = self._build_data_bn(num_person, in_channels, A.size(1))

        num_prototype = kwargs.pop('num_prototype', 100)
        self.stem_kwargs = cp.deepcopy(kwargs)
        self.stem_kwargs.pop('tcn_dropout', None)
        self.stem_kwargs.pop('g1x1', None)
        self.stem_kwargs.pop('gcn_g1x1', None)

        def _expand_stage_kwargs(stage_count):
            stage_kwargs = [cp.deepcopy(kwargs) for _ in range(stage_count)]
            for k, v in kwargs.items():
                if isinstance(v, tuple) and len(v) == stage_count:
                    for i in range(stage_count):
                        stage_kwargs[i][k] = v[i]
            return stage_kwargs

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        if self.multi_branch:
            shared_stage_kwargs = _expand_stage_kwargs(num_stages)
            branch_stage_kwargs = _expand_stage_kwargs(self.multi_branch_stages)
            self.branch_stem = nn.ModuleList([
                GCN_Block(
                    branch_in_channels,
                    base_channels,
                    A.clone(),
                    1,
                    residual=False,
                    use_bottleneck=self.use_bottleneck,
                    **cp.deepcopy(self.stem_kwargs),
                )
                for _ in range(self.branch_num)
            ])
            self.branch_gcn = nn.ModuleList([
                nn.ModuleList([
                    GCN_Block(
                        base_channels,
                        base_channels,
                        A.clone(),
                        1,
                        use_bottleneck=self.use_bottleneck,
                        **branch_stage_kwargs[i],
                    )
                    for i in range(self.multi_branch_stages)
                ])
                for _ in range(self.branch_num)
            ])
            modules = []
            inflate_times = 0
            for i in range(1, num_stages + 1):
                stride = 1 + (i in down_stages)
                in_c = base_channels
                if i in inflate_stages:
                    inflate_times += 1
                out_c = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
                base_channels = out_c
                modules.append(
                    GCN_Block(
                        in_c,
                        out_c,
                        A.clone(),
                        stride,
                        use_bottleneck=self.use_bottleneck,
                        **shared_stage_kwargs[i - 1],
                    )
                )
            self.num_stages = num_stages
            self.gcn = nn.ModuleList(modules)
        else:
            lw_kwargs = _expand_stage_kwargs(num_stages)
            modules = []
            if self.in_channels != self.base_channels:
                modules = [
                    GCN_Block(
                        in_channels,
                        base_channels,
                        A.clone(),
                        1,
                        residual=False,
                        use_bottleneck=self.use_bottleneck,
                        **lw_kwargs[0],
                    )
                ]

            inflate_times = 0
            down_times = 0
            for i in range(2, num_stages + 1):
                stride = 1 + (i in down_stages)
                in_channels = base_channels
                if i in inflate_stages:
                    inflate_times += 1
                out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
                base_channels = out_channels
                modules.append(
                    GCN_Block(
                        in_channels,
                        out_channels,
                        A.clone(),
                        stride,
                        use_bottleneck=self.use_bottleneck,
                        **lw_kwargs[i - 1],
                    )
                )
                down_times += (i in down_stages)

            if self.in_channels == self.base_channels:
                num_stages -= 1

            self.num_stages = num_stages
            self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

        def _infer_graph_channels(block):
            if not hasattr(block, 'gcn'):
                raise AttributeError('Expected GCN_Block to expose a `gcn` module.')
            return block.gcn.num_subsets * block.gcn.mid_channels

        if self.multi_branch:
            graph_channels = _infer_graph_channels(self.gcn[-1])
        else:
            if len(self.gcn) == 0:
                raise RuntimeError('ProtoGCN requires at least one GCN block to build the prototype module.')
            graph_channels = _infer_graph_channels(self.gcn[-1])

        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        
        self.post = nn.Conv2d(graph_channels, graph_channels, 1)
        self.bn = build_norm_layer(norm_cfg, graph_channels)[1]
        self.relu = nn.ReLU()
        
        self.prn = Prototype_Reconstruction_Network(graph_channels, num_prototype)

    def _build_data_bn(self, num_person, channels, num_joints):
        if self.data_bn_type == 'MVC':
            return nn.BatchNorm1d(num_person * channels * num_joints)
        if self.data_bn_type == 'VC':
            return nn.BatchNorm1d(channels * num_joints)
        return nn.Identity()

    def _apply_data_bn(self, x, data_bn, channels):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = data_bn(x.view(N, M * V * channels, T))
        else:
            x = data_bn(x.view(N * M, V * channels, T))
        x = x.view(N, M, V, channels, T).permute(0, 1, 3, 4, 2).contiguous()
        return x.view(N * M, channels, T, V)

    def _run_stage_stack(self, x, modules):
        get_graph = []
        view_logits_list = []
        for module in modules:
            x, gcl_graph = module(x)
            get_graph.append(gcl_graph)
            view_logits = getattr(module.gcn, 'last_view_logits', None)
            if view_logits is not None:
                view_logits_list.append(view_logits)
        return x, get_graph, view_logits_list

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        logger.debug("ProtoGCN.forward: input=%s", _shape(x))
        N, M, T, V, C = x.size()
        if self.multi_branch:
            branch_chunks = torch.chunk(x, self.branch_num, dim=-1)
            branch_outputs = []
            get_graph = []
            view_logits_list = []
            for branch_idx, branch_x in enumerate(branch_chunks):
                branch_x = self._apply_data_bn(
                    branch_x,
                    self.branch_data_bn[branch_idx],
                    self.branch_in_channels,
                )
                logger.debug("ProtoGCN.forward: branch=%d after_data_bn=%s", branch_idx, _shape(branch_x))
                branch_modules = [self.branch_stem[branch_idx], *self.branch_gcn[branch_idx]]
                branch_x, branch_graphs, branch_views = self._run_stage_stack(branch_x, branch_modules)
                branch_outputs.append(branch_x)
                get_graph.extend(branch_graphs)
                view_logits_list.extend(branch_views)
                logger.debug(
                    "ProtoGCN.forward: branch=%d out=%s",
                    branch_idx,
                    _shape(branch_x),
                )
            x = torch.stack(branch_outputs, dim=0).sum(dim=0)
            logger.debug("ProtoGCN.forward: fused_branches=%s", _shape(x))
            x, shared_graphs, shared_views = self._run_stage_stack(x, self.gcn)
            get_graph.extend(shared_graphs)
            view_logits_list.extend(shared_views)
            logger.debug(
                "ProtoGCN.forward: shared_stages=%d out=%s",
                len(shared_graphs),
                _shape(x),
            )
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            if self.data_bn_type == 'MVC':
                x = self.data_bn(x.view(N, M * V * C, T))
            else:
                x = self.data_bn(x.view(N * M, V * C, T))
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
            logger.debug("ProtoGCN.forward: after_data_bn=%s", _shape(x))

            get_graph = []
            view_logits_list = []
            for i in range(self.num_stages):
                x, gcl_graph = self.gcn[i](x)
                # N*M C V V
                get_graph.append(gcl_graph)
                view_logits = getattr(self.gcn[i].gcn, 'last_view_logits', None)
                if view_logits is not None:
                    view_logits_list.append(view_logits)
                logger.debug("ProtoGCN.forward: stage=%d x=%s graph=%s", i, _shape(x), _shape(gcl_graph))
        
        x = x.reshape((N, M) + x.shape[1:])
        logger.debug("ProtoGCN.forward: reshaped_back=%s", _shape(x))

        if len(get_graph) == 0:
            raise RuntimeError('ProtoGCN.forward expected at least one graph tensor, but none was produced.')

        graph = get_graph[-1]
        graph_channels = graph.size(1)
        logger.debug(
            "ProtoGCN.forward: last_graph_raw=%s graph_channels=%d",
            _shape(graph),
            graph_channels,
        )

        # Graph tensors are produced per sample/person as N*M x C x V x V.
        # Pool person dimension first, then flatten the joint-pair dimension.
        graph = graph.view(N, M, graph_channels, V, V).mean(1).reshape(N, graph_channels, V * V)
        logger.debug("ProtoGCN.forward: last_graph_pool=%s", _shape(graph))
        
        the_graph_list = []
        for i in range(N):
            # V*V C
            the_graph = graph[i].permute(1, 0).contiguous()
            # V*V C
            the_graph = self.prn(the_graph)
            # C V V
            the_graph = the_graph.permute(1, 0).contiguous().view(graph_channels, V, V)
            the_graph_list.append(the_graph)
        
        # N C V V
        re_graph = torch.stack(the_graph_list, dim=0)
        re_graph = self.post(re_graph)
        reconstructed_graph = self.relu(self.bn(re_graph))
        # N V*V
        reconstructed_graph = reconstructed_graph.mean(1).view(N, -1)
        logger.debug("ProtoGCN.forward: reconstructed_graph=%s", _shape(reconstructed_graph))

        if len(view_logits_list) > 0:
            view_logits = torch.stack(view_logits_list, dim=0).mean(dim=0)
            view_logits = view_logits.view(N, M, -1).mean(dim=1)
            self.view_logits = view_logits
        else:
            self.view_logits = None
        
        return x, reconstructed_graph
