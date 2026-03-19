# moe_lora.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.drop(self.B(self.A(x))) * self.scaling

class MoELoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        hidden_size: int,
        num_experts=16,
        top_k=2,
        r=8,
        alpha=16,
        dropout=0.05,
        router_aux_weight=0.01
    ):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        # nn.init.zeros_(self.router.weight)
        nn.init.normal_(self.router.weight, std=1e-2)

        self.experts = nn.ModuleList([
            LoRAExpert(self.base.in_features, self.base.out_features, r=r, alpha=alpha, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router_aux_weight = router_aux_weight
        self.latest_aux = torch.tensor(0.0)

    def forward(self, x):
        orig_shape = x.shape
        H = x.shape[-1]
        x2d = x.view(-1, H)

        logits = self.router(x2d)
        gates = F.softmax(logits, dim=-1)                     # [N, E]
        topk_val, topk_idx = torch.topk(gates, k=self.top_k, dim=-1)  # both [N, K]

        # 基座输出
        y_base = self.base(x2d)                                # [N, D_out]
        y_delta = torch.zeros_like(y_base)                     # 增量

        # 将 top-k 门值散射到 [N, E]，其余为 0，便于按专家分组处理
        gates_topk = torch.zeros_like(gates)
        gates_topk.scatter_(1, topk_idx, topk_val)             # [N, E]

        # 仅遍历本批实际被选中的专家，避免全量 E 循环
        active_experts = torch.unique(topk_idx)
        for e_id in active_experts.tolist():                   # 转成 Python int 便于索引 ModuleList
            mask = gates_topk[:, e_id] > 0                     # 该专家被选中的 token
            if mask.any():
                x_sel = x2d[mask]
                w_sel = gates_topk[mask, e_id].unsqueeze(-1)   # [n,1]
                y_sel = self.experts[e_id](x_sel)              # [n,D_out]
                y_delta[mask] += w_sel * y_sel

        y = y_base + y_delta
        y = y.view(*orig_shape[:-1], y.shape[-1])

        load = gates.mean(dim=0)
        # 使用到均匀分布的 KL 形式，数值>=0；注意不要用上文的 token 权重（形状 [N,1]），
        # 否则会使 aux 变为非标量，导致总损失在加上 aux 后不是标量而使 backward 报错。
        kl_uniform = torch.sum(load * torch.log(load + 1e-9)) + math.log(self.num_experts)
        aux = self.router_aux_weight * kl_uniform  # 标量
        self.latest_aux = aux
        return y
