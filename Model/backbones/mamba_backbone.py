# =============================================================================
# Mamba Backbone - 基于Mamba的序列建模主干
# Version: v1.0
#
# Mamba是一种State Space Model (SSM)，具有线性时间复杂度 O(T)
# 对于高频金融数据（T=3000），比Transformer更高效
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# 尝试导入mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Use `pip install mamba-ssm` for Mamba models.")


class MambaBlock(nn.Module):
    """
    单个Mamba块，包含残差连接和LayerNorm
    
    如果mamba-ssm未安装，使用简化的S4-like实现作为替代
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4, 
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True
        else:
            # Fallback: 使用简化的State Space实现
            self.mamba = SimplifiedSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            self.use_mamba = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class SimplifiedSSM(nn.Module):
    """
    简化的State Space Model实现
    
    当mamba-ssm未安装时的替代方案。
    使用因果卷积 + 门控机制模拟SSM的行为。
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        d_inner = d_model * expand
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        
        # 因果卷积
        self.conv = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=d_inner
        )
        
        # 简化的"状态"机制：使用额外的卷积层模拟
        self.state_proj = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_state,
            padding=d_state - 1,
            groups=d_inner
        )
        
        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
        """
        B, T, D = x.shape
        
        # 投影
        xz = self.in_proj(x)  # (B, T, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # 各 (B, T, d_inner)
        
        # 转换为卷积格式
        x_proj = x_proj.transpose(1, 2)  # (B, d_inner, T)
        
        # 因果卷积
        x_conv = self.conv(x_proj)[:, :, :T]  # 截断causal padding
        
        # 状态卷积（模拟SSM的状态传递）
        x_state = self.state_proj(x_conv)[:, :, :T]
        
        # 门控
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)
        x_state = x_state.transpose(1, 2)
        
        # 组合
        x_combined = self.act(x_conv) * x_state + x_conv
        x_combined = x_combined * self.act(z)  # 门控
        
        # 输出投影
        output = self.out_proj(x_combined)
        output = self.dropout(output)
        
        return output


class MambaBackbone(nn.Module):
    """
    基于Mamba的序列建模主干
    
    Mamba具有：
    - 线性时间复杂度 O(T)
    - 长程依赖建模能力
    - 因果性（天然适合时序预测）
    
    对于高频金融数据（T=3000），比Transformer更高效。
    
    Args:
        d_model: 模型维度
        d_state: SSM状态维度
        d_conv: 局部卷积核大小
        expand: 内部扩展因子
        num_layers: Mamba层数
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 堆叠Mamba层
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            
        Returns:
            output: (B, T, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        return x


class HybridMambaTransformer(nn.Module):
    """
    混合Mamba-Transformer架构
    
    - 用Mamba处理长程依赖 (高效)
    - 用少量Transformer层处理关键位置的精细交互
    
    这种混合方式可以在效率和性能之间取得平衡。
    
    Args:
        d_model: 模型维度
        num_mamba_layers: Mamba层数
        num_transformer_layers: Transformer层数
        nhead: 注意力头数
        d_state: SSM状态维度
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_mamba_layers: int = 3,
        num_transformer_layers: int = 1,
        nhead: int = 4,
        d_state: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Mamba处理长程
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(num_mamba_layers)
        ])
        
        # Transformer处理精细交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            causal: 是否使用因果掩码
            
        Returns:
            output: (B, T, d_model)
        """
        # 1. Mamba长程建模
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # 2. Transformer精细交互
        if causal:
            T = x.size(1)
            mask = self._generate_causal_mask(T, x.device)
        else:
            mask = None
            
        x = self.transformer(x, mask=mask)
        x = self.final_norm(x)
        
        return x


class MambaTransformerAlternating(nn.Module):
    """
    交替Mamba-Transformer架构
    
    Mamba和Transformer层交替堆叠：
    Mamba -> Transformer -> Mamba -> Transformer -> ...
    
    这种设计可以在每个尺度上同时利用两者的优势。
    
    Args:
        d_model: 模型维度
        num_blocks: 交替块数量 (每个块包含1个Mamba + 1个Transformer)
        nhead: 注意力头数
        d_state: SSM状态维度
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        d_model: int = 64,
        num_blocks: int = 2,
        nhead: int = 4,
        d_state: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.blocks = nn.ModuleList()
        
        for _ in range(num_blocks):
            # Mamba层
            mamba = MambaBlock(d_model, d_state=d_state, dropout=dropout)
            
            # Transformer层
            transformer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            
            self.blocks.append(nn.ModuleDict({
                'mamba': mamba,
                'transformer': transformer
            }))
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            causal: 是否使用因果掩码
            
        Returns:
            output: (B, T, d_model)
        """
        T = x.size(1)
        mask = self._generate_causal_mask(T, x.device) if causal else None
        
        for block in self.blocks:
            # Mamba
            x = block['mamba'](x)
            # Transformer
            x = block['transformer'](x, src_mask=mask)
        
        x = self.final_norm(x)
        return x
