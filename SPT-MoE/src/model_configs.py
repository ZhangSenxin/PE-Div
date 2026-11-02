import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
SEED = 42
INFERENCE_CONFIG = {'attention_mode': 'expectation', 'allowed_attention_modes': ('expectation', 'sample', 'mc'), 'cublas_workspace_config': ':4096:8', 'batch_size': 256, 'verification_batch_size': 17, 'verification_size': 65, 'batch_atol': 1e-05, 'batch_rtol': 1e-05}
BERNOULLI_VALIDATION_CONFIG = {'sequence_batch_size': 32, 'draw_batch_size': 32}
BERNOULLI_EXECUTION_CONFIG = {'cache_draw_count': 1024}

def set_seed(s=SEED):
    """设置所有随机种子以确保可复现性"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
MAX_L_EVAL = 34

class BST(torch.autograd.Function):
    """伯努利采样 + 直通估计器 (STE)"""

    @staticmethod
    def forward(ctx, p):
        s = torch.bernoulli(p)
        ctx.save_for_backward(p, s)
        return s

    @staticmethod
    def backward(ctx, g):
        return g.clone()

class BHA(nn.Module):
    """伯努利多头注意力

    x: (B, L, D)；key_mask: (B, L) bool，True=有效token（False处的key不参与注意力）
    """

    def __init__(self, ed, nh, dp=0.0):
        super().__init__()
        self.ed, self.nh, self.hd = (ed, nh, ed // nh)
        self.qkv = nn.Linear(ed, ed * 3)
        self.out = nn.Linear(ed, ed)
        self.dp = nn.Dropout(dp)
        self.eval_attention_mode = INFERENCE_CONFIG['attention_mode']

    def forward(self, x, key_mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        a = torch.matmul(q, k.transpose(-2, -1)) / self.hd ** 0.5
        p = torch.sigmoid(a)
        mode = getattr(self, 'eval_attention_mode', INFERENCE_CONFIG['attention_mode'])
        if mode not in INFERENCE_CONFIG['allowed_attention_modes']:
            raise ValueError(f'Unknown attention inference mode: {mode}')
        if not self.training and mode == 'mc':
            uniform = getattr(self, '_mc_uniform', None)
            if uniform is None or uniform.shape != p.shape:
                raise RuntimeError('MC inference requires an explicit sequence/draw sampling context.')
            a = (uniform < p).to(p.dtype)
        else:
            a = p if not self.training and mode == 'expectation' else BST.apply(p)
        if key_mask is not None:
            a = a * key_mask[:, None, None, :].to(a.dtype)
        a = self.dp(a)
        return self.out(torch.matmul(a, v).permute(0, 2, 1, 3).reshape(B, L, D))

def prepare_deterministic_inference():
    """Call before CUDA initialization in a dedicated inference process."""
    setting = INFERENCE_CONFIG['cublas_workspace_config']
    if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != setting:
        raise RuntimeError('Prepare deterministic inference before CUDA initialization.')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = setting
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

class CSA2(nn.Module):
    """CSA2专家模块: 2层Bernoulli注意力 + FFN

    输入兼容两种形态：
    - (B, ind)：定长向量 → 内部变为 L=1（与旧版数值路径一致）
    - (B, L, ind)：不等长序列 token；配合 key_mask=(B,L) 做 masked 池化
    输出均为 (B, hd)
    """

    def __init__(self, ind, hd, outd, dp=0.1, nh=4):
        super().__init__()
        self.fc_in = nn.Linear(ind, hd)
        self.a1 = BHA(hd, nh, dp)
        self.n1 = nn.LayerNorm(hd)
        self.a2 = BHA(hd, nh, dp)
        self.n2 = nn.LayerNorm(hd)
        self.ffn = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Dropout(dp), nn.Linear(hd, hd))
        self.n3 = nn.LayerNorm(hd)

    def forward(self, x, key_mask=None):
        squeeze_l1 = x.dim() == 2
        if squeeze_l1:
            x = x.unsqueeze(1)
            key_mask = None
        h = self.fc_in(x)
        h = self.n1(h + self.a1(h, key_mask))
        h = self.n2(h + self.a2(h, key_mask))
        h = self.n3(h + self.ffn(h))
        if key_mask is not None:
            m = key_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-06)
        if h.size(1) == 1:
            return h.squeeze(1)
        return h.mean(dim=1)

class MoE(nn.Module):
    """混合专家网络

    supp_ind>0 时启用不等长补充特征分支：
    - x:     (B, ind)      主特征（ESM 等定长向量）
    - supp:  (B, L, supp_ind) 不等长序列 token
    - mask:  (B, L) bool   True=真实残基，False=padding
    - supp_fuse_dim: 融合前将 s 投影到的维数（P1.2；None=与 ehd 相同）
    supp_ind=0 时 forward(x) 与旧版完全兼容（无新增参数）。
    """

    def __init__(self, ind, hd, outd, ne, ehd, dp=0.1, nh=4, supp_ind=0, supp_fuse_dim=None):
        super().__init__()
        self.ind = ind
        self.supp_ind = supp_ind
        self.supp_fuse_dim = supp_fuse_dim
        self.experts = nn.ModuleList([CSA2(ind, ehd, outd, dp, nh) for _ in range(ne)])
        self.gating = nn.Sequential(nn.Linear(ind, hd), nn.ReLU(), nn.Linear(hd, ne))
        self.ffn = nn.Sequential(nn.Linear(ehd, ehd), nn.ReLU(), nn.Dropout(dp), nn.Linear(ehd, ehd))
        self.fc_out = nn.Linear(ehd, outd)
        if supp_ind > 0:
            fd = supp_fuse_dim if supp_fuse_dim else ehd
            self.supp_encoder = CSA2(supp_ind, fd, fd, dp, nh)
            self.supp_fuse = nn.Sequential(nn.Linear(ind + fd, ind), nn.ReLU(), nn.Dropout(dp))

    def forward_features(self, x, supp=None, mask=None):
        if self.supp_ind > 0:
            if supp is None:
                raise ValueError(f'MoE(supp_ind={self.supp_ind}) 需要传入 supp (B,L,{self.supp_ind})')
            if supp.dim() != 3:
                raise ValueError(f'supp 应为 (B, L, {self.supp_ind})，收到 {tuple(supp.shape)}')
            if supp.size(-1) != self.supp_ind:
                raise ValueError(f'supp 最后一维应为 {self.supp_ind}，收到 {supp.size(-1)}')
            if mask is not None and mask.shape != supp.shape[:2]:
                raise ValueError(f'mask 应为 {tuple(supp.shape[:2])}，收到 {tuple(mask.shape)}')
            if mask is not None:
                empty = ~mask.any(dim=1)
                if empty.any():
                    fixed = mask.clone()
                    fixed[empty, 0] = True
                    s = self.supp_encoder(supp, fixed)
                    s = s * (~empty).unsqueeze(1).to(s.dtype)
                else:
                    s = self.supp_encoder(supp, mask)
            else:
                s = self.supp_encoder(supp, None)
            x = self.supp_fuse(torch.cat([x, s], dim=-1))
        elif supp is not None:
            raise ValueError('当前 MoE 未启用补充特征分支（supp_ind=0），构造时传 supp_ind>0')
        gw = F.softmax(self.gating(x), dim=-1)
        eo = torch.stack([e(x) for e in self.experts], dim=-1)
        return self.ffn(torch.sum(eo * gw.unsqueeze(1), dim=-1))

    def forward(self, x, supp=None, mask=None):
        return self.fc_out(self.forward_features(x, supp, mask))

class Phase6TaskMoE(MoE):
    """Existing MoE backbone, retained DPF head, DMS head and optional regressor."""

    def __init__(self, cfg, role):
        super().__init__(ind=cfg['in_dim'], hd=cfg['hidden_dim'], outd=cfg['out_dim'], ne=cfg['num_experts'], ehd=cfg['expert_hidden_dim'], dp=cfg['dropout'], nh=cfg['num_heads'], supp_ind=cfg['supp_ind'], supp_fuse_dim=cfg['fuse_dim'])
        self.phase6_role = role
        self.target_head = nn.Linear(cfg['expert_hidden_dim'], cfg['out_dim'])
        self.target_head.load_state_dict(self.fc_out.state_dict())

    def forward_task_margins(self, x, supp=None, mask=None):
        hidden = self.forward_features(x, supp, mask)
        p = self.fc_out(hidden)
        dpf = p[:, 1] - p[:, 0]
        q = self.target_head(hidden)
        dms = dpf if self.phase6_role == 'P' else q[:, 1] - q[:, 0]
        reg = dms
        unified = dms
        return torch.stack([dms, dpf, reg, unified], dim=1)

    def forward(self, x, supp=None, mask=None):
        margin = self.forward_task_margins(x, supp, mask)[:, 0]
        return torch.stack([torch.zeros_like(margin), margin], dim=1)
