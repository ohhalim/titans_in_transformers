'''
==================================================
테스트: Memory as a Context (MAC)
==================================================
입력 크기: torch.Size([2, 512, 512])
출력 크기: torch.Size([2, 512, 512])
메모리 상태 크기: torch.Size([512, 512])

==================================================
테스트: Memory as a Gate (MAG)
==================================================
입력 크기: torch.Size([2, 512, 512])
출력 크기: torch.Size([2, 512, 512])
메모리 상태 크기: torch.Size([512, 512])

==================================================
테스트: Memory as a Layer (MAL)
==================================================
입력 크기: torch.Size([2, 512, 512])
출력 크기: torch.Size([2, 512, 512])
메모리 상태 크기: torch.Size([512, 512])

==================================================
모델 출력 비교 (L2 norm)
==================================================
MAC vs MAG: 597.7224
MAC vs MAL: 594.8182
MAG vs MAL: 600.1712
(base) (venv) ohhalim@MacBookAir titans_in_transformers % python Titans.py

--- MAC 모델 테스트 ---
입력 크기: torch.Size([2, 512, 512])
출력 크기: torch.Size([2, 512, 512])
메모리 상태 크기: torch.Size([512, 512])

--- MAG 모델 테스트 ---
입력 크기: torch.Size([2, 512, 512])
출력 크기: torch.Size([2, 512, 512])
메모리 상태 크기: torch.Size([512, 512])

--- MAL 모델 테스트 ---
입력 크기: torch.Size([2, 512, 512])
출력 크기: torch.Size([2, 512, 512])
메모리 상태 크기: torch.Size([512, 512])
(base) (venv) ohhalim@MacBookAir titans_in_transformers % 
'''                          


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List

# ------------------ 메모리 모듈 ------------------

class LongTermMemory(nn.Module):
    """
    신경망 장기 메모리 모듈 (Neural Long-term Memory)
    테스트 타임에 메모리를 학습하는 메타 모델로 작동
    """
    def __init__(
        self, 
        dim: int, 
        memory_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.memory_depth = memory_depth
        
        # 메모리 모듈 (MLP)
        layers = []
        for i in range(memory_depth):
            if i == 0:
                layers.append(nn.Linear(dim, dim))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(dim, dim))
                if i < memory_depth - 1:
                    layers.append(nn.SiLU())
                    layers.append(nn.Dropout(dropout))
        
        self.memory_network = nn.Sequential(*layers)
        
        # 망각 게이트와 모멘텀을 위한 파라미터
        self.alpha_proj = nn.Linear(dim, 1)  # 망각 게이트
        self.eta_proj = nn.Linear(dim, 1)    # 과거 놀라움 계수
        self.theta_proj = nn.Linear(dim, 1)  # 순간 놀라움 계수
        
        # 각 파라미터에 대한 모멘텀 값 초기화
        self.momentum = {}
        for name, param in self.memory_network.named_parameters():
            self.momentum[name] = torch.zeros_like(param.data)
        
    def compute_gates(self, x: torch.Tensor) -> Tuple[float, float, float]:
        """입력에 기반한 망각 게이트와 놀라움 계수 계산"""
        # 배치 평균값 사용하여 단일 스칼라 값 반환
        alpha = torch.sigmoid(self.alpha_proj(x.mean(dim=0, keepdim=True))).mean().item()
        eta = torch.sigmoid(self.eta_proj(x.mean(dim=0, keepdim=True))).mean().item()
        theta = F.softplus(self.theta_proj(x.mean(dim=0, keepdim=True))).mean().item()
        
        return alpha, eta, theta
    
    def compute_surprise(
        self, 
        x: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> float:
        """
        입력에 대한 놀라움 측정
        놀라움 = 현재 메모리로 예측한 값과 실제 값의 차이
        """
        with torch.no_grad():
            pred_values = self.memory_network(keys)
            surprise = F.mse_loss(pred_values, values)
            
        return surprise.item()
    
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        메모리에서 정보 검색 및 업데이트
        
        Args:
            query: 메모리에서 검색할 쿼리
            key: 메모리에 저장할 키 (없으면 쿼리를 사용)
            value: 메모리에 저장할 값 (없으면 쿼리를 사용)
            update_memory: 메모리 업데이트 여부
        
        Returns:
            메모리에서 검색된 정보
        """
        if key is None:
            key = query
        
        if value is None:
            value = query
            
        # 1. 메모리 내용 검색 (가중치 업데이트 없이)
        output = self.memory_network(query)
        
        # 메모리 업데이트가 필요하지 않으면 여기서 종료
        if not update_memory or not self.training:
            return output
        
        # 2. 메모리 업데이트 (테스트 타임 학습)
        # 망각 게이트와 놀라움 계수 계산 (스칼라 값 사용)
        alpha, eta, theta = self.compute_gates(key)
        
        # 놀라움 계산 - 현재 메모리가 얼마나 잘 예측하는지 측정
        momentary_surprise = self.compute_surprise(key, key, value)
        momentary_surprise = momentary_surprise * theta
        
        # 메모리 업데이트를 위한 그라디언트 계산 (매뉴얼 모드)
        with torch.enable_grad():
            pred_value = self.memory_network(key)
            loss = F.mse_loss(pred_value, value)
            grads = torch.autograd.grad(loss, self.memory_network.parameters())
            
            # 그라디언트로 메모리 업데이트
            with torch.no_grad():
                for i, (name, p) in enumerate(self.memory_network.named_parameters()):
                    g = grads[i]
                    
                    # 과거 놀라움과 현재 놀라움 결합 (모멘텀 업데이트)
                    self.momentum[name] = eta * self.momentum[name] - theta * g
                    
                    # 망각 게이트 적용 (weight decay) 및 놀라움 기반 업데이트
                    p.data = (1 - alpha) * p.data + self.momentum[name]
                    
        return output


class PersistentMemory(nn.Module):
    """데이터 독립적인 영구 메모리 모듈"""
    def __init__(self, dim: int, num_memories: int = 16):
        super().__init__()
        self.memories = nn.Parameter(torch.randn(num_memories, dim) * 0.02)
        self.num_memories = num_memories
        
    def forward(self, batch_size: int) -> torch.Tensor:
        # 배치 크기에 맞게 영구 메모리 복제
        return self.memories.unsqueeze(0).expand(batch_size, -1, -1)


# ------------------ 어텐션 모듈 ------------------

class SlidingWindowAttention(nn.Module):
    """슬라이딩 윈도우 어텐션 (단기 메모리용)"""
    def __init__(
        self, 
        dim: int, 
        window_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 슬라이딩 윈도우 어텐션: 각 토큰은 자신과 이전 window_size-1개 토큰만 볼 수 있음
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 스케일링 팩터
        scaling = self.head_dim ** -0.5
        
        # 어텐션 계산 (슬라이딩 윈도우 적용)
        attn_output = []
        
        for i in range(seq_len):
            # 현재 토큰의 윈도우 범위 결정
            start_idx = max(0, i - self.window_size + 1)
            
            # 현재 윈도우에 대한 어텐션 계산
            q_i = q[:, :, i:i+1]  # (batch, heads, 1, head_dim)
            k_window = k[:, :, start_idx:i+1]  # (batch, heads, window, head_dim)
            v_window = v[:, :, start_idx:i+1]  # (batch, heads, window, head_dim)
            
            # 어텐션 스코어 계산
            attn_scores = torch.matmul(q_i, k_window.transpose(-1, -2)) * scaling  # (batch, heads, 1, window)
            
            # 마스킹 적용 (필요한 경우)
            if mask is not None:
                window_mask = mask[:, :, i:i+1, start_idx:i+1]
                attn_scores = attn_scores.masked_fill(window_mask == 0, -1e9)
            
            # 소프트맥스 및 드롭아웃
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 어텐션 출력 계산
            out_i = torch.matmul(attn_weights, v_window)  # (batch, heads, 1, head_dim)
            attn_output.append(out_i)
            
        # 결과 결합
        attn_output = torch.cat(attn_output, dim=2)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # 출력 프로젝션
        output = self.o_proj(attn_output)
        
        return output


# ------------------ Titans 모델 기본 클래스 ------------------

class TitansBase(nn.Module):
    """Titans 모델의 기본 클래스"""
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        memory_depth: int = 2,
        window_size: int = 512,
        persistent_memories: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # 공통 모듈
        self.embedding = nn.Linear(dim, dim)
        self.persistent_memory = PersistentMemory(dim, persistent_memories)
        self.long_term_memory = LongTermMemory(dim, memory_depth, dropout)
        
        # 프로젝션 레이어
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 출력 층
        self.output_norm = nn.LayerNorm(dim)
        self.output_layer = nn.Linear(dim, dim)
        
    def forward(
        self, 
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")


# ------------------ Titans 모델 구현 ------------------

class TitansMAC(TitansBase):
    """
    Titans - Memory as a Context (MAC) 구현
    
    장기 메모리와 영구 메모리를 컨텍스트로 사용하는 아키텍처
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        memory_depth: int = 2,
        window_size: int = 512,
        segment_size: int = 256,
        persistent_memories: int = 16,
        dropout: float = 0.1
    ):
        super().__init__(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            memory_depth=memory_depth,
            window_size=window_size,
            persistent_memories=persistent_memories,
            dropout=dropout
        )
        self.segment_size = segment_size
        
        # 어텐션 모듈 (단기 메모리)
        self.attention_layers = nn.ModuleList([
            SlidingWindowAttention(dim, window_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 레이어 정규화
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(num_layers)
        ])
        
        # 순방향 네트워크
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # 입력 임베딩
        x = self.embedding(x)
        
        # 시퀀스를 세그먼트로 분할 (논문에서 제안한 세그먼트 기반 처리)
        segments = []
        segment_outputs = []
        
        for i in range(0, seq_len, self.segment_size):
            end_idx = min(i + self.segment_size, seq_len)
            segment = x[:, i:end_idx]
            segments.append(segment)
            
            # 영구 메모리 가져오기
            persistent_mem = self.persistent_memory(batch_size)
            
            # 현재 세그먼트에 대한 쿼리, 키, 값 생성
            query = self.q_proj(segment)
            key = self.k_proj(segment)
            value = self.v_proj(segment)
            
            # 장기 메모리에서 관련 정보 검색
            if memory_state is not None:
                # 이전 메모리 상태 활용
                memory_output = self.long_term_memory(query, update_memory=False)
            else:
                # 초기 상태
                memory_output = torch.zeros_like(query)
            
            # 컨텍스트 생성: 영구 메모리 + 장기 메모리 + 현재 세그먼트
            context = torch.cat([persistent_mem, memory_output, segment], dim=1)
            
            # 어텐션 레이어 통과 (단기 메모리)
            for j, (attn, norm, ffn) in enumerate(zip(self.attention_layers, self.norm_layers, self.ffn_layers)):
                # 어텐션 계산
                context_norm = norm(context)
                context_attn = attn(context_norm)
                context = context + context_attn
                
                # FFN 통과
                context_norm = norm(context)
                context_ffn = ffn(context_norm)
                context = context + context_ffn
            
            # 최종 출력 계산
            segment_output = self.output_layer(self.output_norm(context[:, persistent_mem.size(1) + memory_output.size(1):]))
            segment_outputs.append(segment_output)
            
            # 장기 메모리 업데이트
            self.long_term_memory(query, key, value, update_memory=True)
            
            # 메모리 상태 업데이트
            memory_state = self.long_term_memory.memory_network[-1].weight.clone()
            
        # 모든 세그먼트 출력 결합
        output = torch.cat(segment_outputs, dim=1)
        
        return output, memory_state


class TitansMAG(TitansBase):
    """
    Titans - Memory as a Gate (MAG) 구현
    
    장기 메모리와 단기 메모리(어텐션)를 게이팅으로 결합하는 아키텍처
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        memory_depth: int = 2,
        window_size: int = 512,
        persistent_memories: int = 16,
        dropout: float = 0.1
    ):
        super().__init__(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            memory_depth=memory_depth,
            window_size=window_size,
            persistent_memories=persistent_memories,
            dropout=dropout
        )
        
        # 슬라이딩 윈도우 어텐션 (단기 메모리)
        self.attention = SlidingWindowAttention(dim, window_size, num_heads, dropout)
        
        # 게이팅 메커니즘
        self.gate_norm1 = nn.LayerNorm(dim)
        self.gate_norm2 = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        
        # 레이어 정규화 및 FFN
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                SlidingWindowAttention(dim, window_size, num_heads, dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout)
                )
            ])
            for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # 입력 임베딩
        x = self.embedding(x)
        
        # 영구 메모리 가져와서 입력에 결합
        persistent_mem = self.persistent_memory(batch_size)
        x_with_persistent = torch.cat([persistent_mem, x], dim=1)
        
        # 프로젝션
        query = self.q_proj(x_with_persistent)
        key = self.k_proj(x_with_persistent)
        value = self.v_proj(x_with_persistent)
        
        # 1. 슬라이딩 윈도우 어텐션 (단기 메모리)
        attention_output = self.attention(x_with_persistent)
        
        # 2. 장기 메모리
        memory_output = self.long_term_memory(query, key, value, update_memory=True)
        
        # 메모리 상태 업데이트
        memory_state = self.long_term_memory.memory_network[-1].weight.clone()
        
        # 3. 게이팅 메커니즘으로 두 메모리 결합
        attn_norm = self.gate_norm1(attention_output)
        mem_norm = self.gate_norm2(memory_output)
        
        # 게이트 계산
        gate_input = torch.cat([attn_norm, mem_norm], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        # 게이팅으로 결합
        combined = gate * attn_norm + (1 - gate) * mem_norm
        
        # 레이어 통과
        x = combined
        for layer_norm1, attn, layer_norm2, ffn in self.layers:
            # 어텐션 레이어
            x_norm = layer_norm1(x)
            x = x + attn(x_norm)
            
            # FFN 레이어
            x_norm = layer_norm2(x)
            x = x + ffn(x_norm)
        
        # 영구 메모리 제외한 부분만 추출
        output = x[:, persistent_mem.size(1):]
        
        # 출력 층
        output = self.output_layer(self.output_norm(output))
        
        return output, memory_state


class TitansMAL(TitansBase):
    """
    Titans - Memory as a Layer (MAL) 구현
    
    장기 메모리를 레이어로 사용하는 아키텍처
    장기 메모리 -> 어텐션의 순차적인 구조
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        memory_depth: int = 2,
        window_size: int = 512,
        persistent_memories: int = 16,
        dropout: float = 0.1
    ):
        super().__init__(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            memory_depth=memory_depth,
            window_size=window_size,
            persistent_memories=persistent_memories,
            dropout=dropout
        )
        
        # 슬라이딩 윈도우 어텐션 레이어 (단기 메모리)
        self.attention_layers = nn.ModuleList([
            SlidingWindowAttention(dim, window_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 레이어 정규화
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(dim)
            for _ in range(num_layers + 1)  # +1 for memory output norm
        ])
        
        # 순방향 네트워크
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # 입력 임베딩
        x = self.embedding(x)
        
        # 영구 메모리 가져와서 입력에 결합
        persistent_mem = self.persistent_memory(batch_size)
        x_with_persistent = torch.cat([persistent_mem, x], dim=1)
        
        # 프로젝션
        query = self.q_proj(x_with_persistent)
        key = self.k_proj(x_with_persistent)
        value = self.v_proj(x_with_persistent)
        
        # 1. 장기 메모리 레이어 (첫 번째 레이어로 처리)
        memory_output = self.long_term_memory(query, key, value, update_memory=True)
        
        # 메모리 상태 업데이트
        memory_state = self.long_term_memory.memory_network[-1].weight.clone()
        
        # 메모리 출력 정규화
        memory_output = self.norm_layers[0](memory_output)
        
        # 2. 어텐션 레이어 + FFN으로 순차적 처리
        x = memory_output
        for i in range(self.num_layers):
            # 어텐션 레이어
            residual = x
            x = self.norm_layers[i+1](x)
            x = self.attention_layers[i](x)
            x = residual + x
            
            # FFN 레이어
            residual = x
            x = self.norm_layers[i+1](x)
            x = self.ffn_layers[i](x)
            x = residual + x
        
        # 영구 메모리 제외한 부분만 추출
        output = x[:, persistent_mem.size(1):]
        
        # 출력 층
        output = self.output_layer(self.output_norm(output))
        
        return output, memory_state


# ------------------ 사용 예시 ------------------

def example_usage():
    # 모델 초기화 (3가지 아키텍처 중 선택)
    models = {
        'MAC': TitansMAC(
            dim=512,
            num_layers=4,
            num_heads=8,
            memory_depth=2,
            window_size=256,
            segment_size=128,
            persistent_memories=16,
            dropout=0.1
        ),
        'MAG': TitansMAG(
            dim=512,
            num_layers=4,
            num_heads=8,
            memory_depth=2,
            window_size=256,
            persistent_memories=16,
            dropout=0.1
        ),
        'MAL': TitansMAL(
            dim=512,
            num_layers=4,
            num_heads=8,
            memory_depth=2,
            window_size=256,
            persistent_memories=16,
            dropout=0.1
        )
    }
    
    # 입력 데이터 (배치 크기 2, 시퀀스 길이 512, 차원 512)
    x = torch.randn(2, 512, 512)
    
    # 각 모델 추론 및 결과 출력
    for name, model in models.items():
        print(f"\n--- {name} 모델 테스트 ---")
        output, memory_state = model(x)
        print(f"입력 크기: {x.shape}")
        print(f"출력 크기: {output.shape}")
        print(f"메모리 상태 크기: {memory_state.shape}")
    
    return models

if __name__ == "__main__":
    example_usage()



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Optional, Tuple, Union, List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple

# class LongTermMemory(nn.Module):
#     """
#     신경망 장기 메모리 모듈 (Neural Long-term Memory)
#     테스트 타임에 메모리를 학습하는 메타 모델로 작동
#     """
#     def __init__(
#         self, 
#         dim: int, 
#         memory_depth: int = 2,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.memory_depth = memory_depth
        
#         # 메모리 모듈 (MLP)
#         layers = []
#         for i in range(memory_depth):
#             if i == 0:
#                 layers.append(nn.Linear(dim, dim))
#                 layers.append(nn.SiLU())
#                 layers.append(nn.Dropout(dropout))
#             else:
#                 layers.append(nn.Linear(dim, dim))
#                 if i < memory_depth - 1:
#                     layers.append(nn.SiLU())
#                     layers.append(nn.Dropout(dropout))
        
#         self.memory_network = nn.Sequential(*layers)
        
#         # 망각 게이트와 모멘텀을 위한 파라미터
#         self.alpha_proj = nn.Linear(dim, 1)  # 망각 게이트
#         self.eta_proj = nn.Linear(dim, 1)    # 과거 놀라움 계수
#         self.theta_proj = nn.Linear(dim, 1)  # 순간 놀라움 계수
        
#         # 각 파라미터에 대한 모멘텀 값 초기화
#         self.momentum = {}
#         for name, param in self.memory_network.named_parameters():
#             self.momentum[name] = torch.zeros_like(param.data)
        
#     def compute_gates(self, x: torch.Tensor) -> Tuple[float, float, float]:
#         """입력에 기반한 망각 게이트와 놀라움 계수 계산"""
#         # 배치 평균값 사용하여 단일 스칼라 값 반환
#         alpha = torch.sigmoid(self.alpha_proj(x.mean(dim=0, keepdim=True))).mean().item()
#         eta = torch.sigmoid(self.eta_proj(x.mean(dim=0, keepdim=True))).mean().item()
#         theta = F.softplus(self.theta_proj(x.mean(dim=0, keepdim=True))).mean().item()
        
#         return alpha, eta, theta
    
#     def compute_surprise(
#         self, 
#         x: torch.Tensor, 
#         keys: torch.Tensor, 
#         values: torch.Tensor
#     ) -> float:
#         """
#         입력에 대한 놀라움 측정
#         놀라움 = 현재 메모리로 예측한 값과 실제 값의 차이
#         """
#         with torch.no_grad():
#             pred_values = self.memory_network(keys)
#             surprise = F.mse_loss(pred_values, values)
            
#         return surprise.item()
    
#     def forward(
#         self, 
#         query: torch.Tensor,
#         key: Optional[torch.Tensor] = None,
#         value: Optional[torch.Tensor] = None,
#         update_memory: bool = True
#     ) -> torch.Tensor:
#         """
#         메모리에서 정보 검색 및 업데이트
        
#         Args:
#             query: 메모리에서 검색할 쿼리
#             key: 메모리에 저장할 키 (없으면 쿼리를 사용)
#             value: 메모리에 저장할 값 (없으면 쿼리를 사용)
#             update_memory: 메모리 업데이트 여부
        
#         Returns:
#             메모리에서 검색된 정보
#         """
#         if key is None:
#             key = query
        
#         if value is None:
#             value = query
            
#         # 1. 메모리 내용 검색 (가중치 업데이트 없이)
#         output = self.memory_network(query)
        
#         # 메모리 업데이트가 필요하지 않으면 여기서 종료
#         if not update_memory or not self.training:
#             return output
        
#         # 2. 메모리 업데이트 (테스트 타임 학습)
#         # 망각 게이트와 놀라움 계수 계산 (스칼라 값 사용)
#         alpha, eta, theta = self.compute_gates(key)
        
#         # 놀라움 계산 - 현재 메모리가 얼마나 잘 예측하는지 측정
#         momentary_surprise = self.compute_surprise(key, key, value)
#         momentary_surprise = momentary_surprise * theta
        
#         # 메모리 업데이트를 위한 그라디언트 계산 (매뉴얼 모드)
#         with torch.enable_grad():
#             pred_value = self.memory_network(key)
#             loss = F.mse_loss(pred_value, value)
#             grads = torch.autograd.grad(loss, self.memory_network.parameters())
            
#             # 그라디언트로 메모리 업데이트
#             with torch.no_grad():
#                 for i, (name, p) in enumerate(self.memory_network.named_parameters()):
#                     g = grads[i]
                    
#                     # 과거 놀라움과 현재 놀라움 결합 (모멘텀 업데이트)
#                     self.momentum[name] = eta * self.momentum[name] - theta * g
                    
#                     # 망각 게이트 적용 (weight decay) 및 놀라움 기반 업데이트
#                     p.data = (1 - alpha) * p.data + self.momentum[name]
                    
#         return output
    
# class PersistentMemory(nn.Module):
#     """데이터 독립적인 영구 메모리 모듈"""
#     def __init__(self, dim: int, num_memories: int = 16):
#         super().__init__()
#         self.memories = nn.Parameter(torch.randn(num_memories, dim) * 0.02)
#         self.num_memories = num_memories
        
#     def forward(self, batch_size: int) -> torch.Tensor:
#         # 배치 크기에 맞게 영구 메모리 복제
#         return self.memories.unsqueeze(0).expand(batch_size, -1, -1)

# class SlidingWindowAttention(nn.Module):
#     """슬라이딩 윈도우 어텐션 (단기 메모리용)"""
#     def __init__(
#         self, 
#         dim: int, 
#         window_size: int = 512,
#         num_heads: int = 8,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
        
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.o_proj = nn.Linear(dim, dim)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         batch_size, seq_len, _ = x.shape
        
#         # 슬라이딩 윈도우 어텐션: 각 토큰은 자신과 이전 window_size-1개 토큰만 볼 수 있음
#         q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
#         # 스케일링 팩터
#         scaling = self.head_dim ** -0.5
        
#         # 어텐션 계산 (슬라이딩 윈도우 적용)
#         attn_output = []
        
#         for i in range(seq_len):
#             # 현재 토큰의 윈도우 범위 결정
#             start_idx = max(0, i - self.window_size + 1)
            
#             # 현재 윈도우에 대한 어텐션 계산
#             q_i = q[:, :, i:i+1]  # (batch, heads, 1, head_dim)
#             k_window = k[:, :, start_idx:i+1]  # (batch, heads, window, head_dim)
#             v_window = v[:, :, start_idx:i+1]  # (batch, heads, window, head_dim)
            
#             # 어텐션 스코어 계산
#             attn_scores = torch.matmul(q_i, k_window.transpose(-1, -2)) * scaling  # (batch, heads, 1, window)
            
#             # 마스킹 적용 (필요한 경우)
#             if mask is not None:
#                 window_mask = mask[:, :, i:i+1, start_idx:i+1]
#                 attn_scores = attn_scores.masked_fill(window_mask == 0, -1e9)
            
#             # 소프트맥스 및 드롭아웃
#             attn_weights = F.softmax(attn_scores, dim=-1)
#             attn_weights = self.dropout(attn_weights)
            
#             # 어텐션 출력 계산
#             out_i = torch.matmul(attn_weights, v_window)  # (batch, heads, 1, head_dim)
#             attn_output.append(out_i)
            
#         # 결과 결합
#         attn_output = torch.cat(attn_output, dim=2)  # (batch, heads, seq_len, head_dim)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
#         # 출력 프로젝션
#         output = self.o_proj(attn_output)
        
#         return output

# class TitansMAC(nn.Module):
#     """
#     Titans - Memory as a Context (MAC) 구현
    
#     장기 메모리와 영구 메모리를 컨텍스트로 사용하는 아키텍처
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_layers: int = 4,
#         num_heads: int = 8,
#         memory_depth: int = 2,
#         window_size: int = 512,
#         segment_size: int = 256,
#         persistent_memories: int = 16,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.dim = dim
#         self.segment_size = segment_size
#         self.window_size = window_size
        
#         # 입력 임베딩
#         self.embedding = nn.Linear(dim, dim)
        
#         # 영구 메모리 (태스크 지식)
#         self.persistent_memory = PersistentMemory(dim, persistent_memories)
        
#         # 장기 메모리 모듈
#         self.long_term_memory = LongTermMemory(dim, memory_depth, dropout)
        
#         # 어텐션 모듈 (단기 메모리)
#         self.attention_layers = nn.ModuleList([
#             SlidingWindowAttention(dim, window_size, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
        
#         # 레이어 정규화
#         self.norm_layers = nn.ModuleList([
#             nn.LayerNorm(dim)
#             for _ in range(num_layers)
#         ])
        
#         # 순방향 네트워크
#         self.ffn_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, dim * 4),
#                 nn.SiLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(dim * 4, dim),
#                 nn.Dropout(dropout)
#             )
#             for _ in range(num_layers)
#         ])
        
#         # 출력 층
#         self.output_norm = nn.LayerNorm(dim)
#         self.output_layer = nn.Linear(dim, dim)
        
#         # 프로젝션 레이어
#         self.q_proj = nn.Linear(dim, dim)  # 쿼리 프로젝션
#         self.k_proj = nn.Linear(dim, dim)  # 키 프로젝션
#         self.v_proj = nn.Linear(dim, dim)  # 값 프로젝션
        
#     def forward(
#         self, 
#         x: torch.Tensor,
#         memory_state: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len, _ = x.shape
        
#         # 입력 임베딩
#         x = self.embedding(x)
        
#         # 시퀀스를 세그먼트로 분할 (논문에서 제안한 세그먼트 기반 처리)
#         segments = []
#         segment_outputs = []
        
#         for i in range(0, seq_len, self.segment_size):
#             end_idx = min(i + self.segment_size, seq_len)
#             segment = x[:, i:end_idx]
#             segments.append(segment)
            
#             # 영구 메모리 가져오기
#             persistent_mem = self.persistent_memory(batch_size)
            
#             # 현재 세그먼트에 대한 쿼리, 키, 값 생성
#             query = self.q_proj(segment)
#             key = self.k_proj(segment)
#             value = self.v_proj(segment)
            
#             # 장기 메모리에서 관련 정보 검색
#             if memory_state is not None:
#                 # 이전 메모리 상태 활용
#                 memory_output = self.long_term_memory(query, update_memory=False)
#             else:
#                 # 초기 상태
#                 memory_output = torch.zeros_like(query)
            
#             # 컨텍스트 생성: 영구 메모리 + 장기 메모리 + 현재 세그먼트
#             context = torch.cat([persistent_mem, memory_output, segment], dim=1)
            
#             # 어텐션 레이어 통과 (단기 메모리)
#             for j, (attn, norm, ffn) in enumerate(zip(self.attention_layers, self.norm_layers, self.ffn_layers)):
#                 # 어텐션 계산
#                 context_norm = norm(context)
#                 context_attn = attn(context_norm)
#                 context = context + context_attn
                
#                 # FFN 통과
#                 context_norm = norm(context)
#                 context_ffn = ffn(context_norm)
#                 context = context + context_ffn
            
#             # 최종 출력 계산
#             segment_output = self.output_layer(self.output_norm(context[:, persistent_mem.size(1) + memory_output.size(1):]))
#             segment_outputs.append(segment_output)
            
#             # 장기 메모리 업데이트
#             self.long_term_memory(query, key, value, update_memory=True)
            
#             # 메모리 상태 업데이트
#             memory_state = self.long_term_memory.memory_network[-1].weight.clone()
            
#         # 모든 세그먼트 출력 결합
#         output = torch.cat(segment_outputs, dim=1)
        
#         return output, memory_state

# class TitansMAG(nn.Module):
#     """
#     Titans - Memory as a Gate (MAG) 구현
    
#     장기 메모리와 단기 메모리(어텐션)를 게이팅으로 결합하는 아키텍처
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_layers: int = 4,
#         num_heads: int = 8,
#         memory_depth: int = 2,
#         window_size: int = 512,
#         persistent_memories: int = 16,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_layers = num_layers 
        
#         # 입력 임베딩
#         self.embedding = nn.Linear(dim, dim)
        
#         # 영구 메모리 (태스크 지식)
#         self.persistent_memory = PersistentMemory(dim, persistent_memories)
        
#         # 장기 메모리 모듈
#         self.long_term_memory = LongTermMemory(dim, memory_depth, dropout)
        
#         # 슬라이딩 윈도우 어텐션 (단기 메모리)
#         self.attention = SlidingWindowAttention(dim, window_size, num_heads, dropout)
        
#         # 게이팅 메커니즘
#         self.gate_norm1 = nn.LayerNorm(dim)
#         self.gate_norm2 = nn.LayerNorm(dim)
#         self.gate_proj = nn.Linear(dim * 2, dim)
        
#         # 레이어 정규화 및 FFN
#         self.layers = nn.ModuleList([
#             nn.ModuleList([
#                 nn.LayerNorm(dim),
#                 SlidingWindowAttention(dim, window_size, num_heads, dropout),
#                 nn.LayerNorm(dim),
#                 nn.Sequential(
#                     nn.Linear(dim, dim * 4),
#                     nn.SiLU(),
#                     nn.Dropout(dropout),
#                     nn.Linear(dim * 4, dim),
#                     nn.Dropout(dropout)
#                 )
#             ])
#             for _ in range(num_layers)
#         ])
        
#         # 출력 층
#         self.output_norm = nn.LayerNorm(dim)
#         self.output_layer = nn.Linear(dim, dim)
        
#         # 프로젝션 레이어
#         self.q_proj = nn.Linear(dim, dim)  # 쿼리 프로젝션
#         self.k_proj = nn.Linear(dim, dim)  # 키 프로젝션
#         self.v_proj = nn.Linear(dim, dim)  # 값 프로젝션
        
#     def forward(
#         self, 
#         x: torch.Tensor,
#         memory_state: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len, _ = x.shape
        
#         # 입력 임베딩
#         x = self.embedding(x)
        
#         # 영구 메모리 가져와서 입력에 결합
#         persistent_mem = self.persistent_memory(batch_size)
#         x_with_persistent = torch.cat([persistent_mem, x], dim=1)
        
#         # 프로젝션
#         query = self.q_proj(x_with_persistent)
#         key = self.k_proj(x_with_persistent)
#         value = self.v_proj(x_with_persistent)
        
#         # 1. 슬라이딩 윈도우 어텐션 (단기 메모리)
#         attention_output = self.attention(x_with_persistent)
        
#         # 2. 장기 메모리
#         memory_output = self.long_term_memory(query, key, value, update_memory=True)
        
#         # 메모리 상태 업데이트
#         memory_state = self.long_term_memory.memory_network[-1].weight.clone()
        
#         # 3. 게이팅 메커니즘으로 두 메모리 결합
#         attn_norm = self.gate_norm1(attention_output)
#         mem_norm = self.gate_norm2(memory_output)
        
#         # 게이트 계산
#         gate_input = torch.cat([attn_norm, mem_norm], dim=-1)
#         gate = torch.sigmoid(self.gate_proj(gate_input))
        
#         # 게이팅으로 결합
#         combined = gate * attn_norm + (1 - gate) * mem_norm
        
#         # 레이어 통과
#         x = combined
#         for layer_norm1, attn, layer_norm2, ffn in self.layers:
#             # 어텐션 레이어
#             x_norm = layer_norm1(x)
#             x = x + attn(x_norm)
            
#             # FFN 레이어
#             x_norm = layer_norm2(x)
#             x = x + ffn(x_norm)
        
#         # 영구 메모리 제외한 부분만 추출
#         output = x[:, persistent_mem.size(1):]
        
#         # 출력 층
#         output = self.output_layer(self.output_norm(output))
        
#         return output, memory_state
# class TitansMAL(nn.Module):
#     """
#     Titans - Memory as a Layer (MAL) 구현
    
#     장기 메모리를 레이어로 사용하는 아키텍처
#     장기 메모리 -> 어텐션의 순차적인 구조
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_layers: int = 4,
#         num_heads: int = 8,
#         memory_depth: int = 2,
#         window_size: int = 512,
#         persistent_memories: int = 16,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.dim = dim
#         self.num_layers = num_layers  # 이 변수를 저장하여 forward 메서드에서 사용
        
#         # 입력 임베딩
#         self.embedding = nn.Linear(dim, dim)
        
#         # 영구 메모리 (태스크 지식)
#         self.persistent_memory = PersistentMemory(dim, persistent_memories)
        
#         # 장기 메모리 모듈
#         self.long_term_memory = LongTermMemory(dim, memory_depth, dropout)
        
#         # 슬라이딩 윈도우 어텐션 레이어 (단기 메모리)
#         self.attention_layers = nn.ModuleList([
#             SlidingWindowAttention(dim, window_size, num_heads, dropout)
#             for _ in range(num_layers)
#         ])
        
#         # 레이어 정규화
#         self.norm_layers = nn.ModuleList([
#             nn.LayerNorm(dim)
#             for _ in range(num_layers + 1)  # +1 for memory output norm
#         ])
        
#         # 순방향 네트워크
#         self.ffn_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, dim * 4),
#                 nn.SiLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(dim * 4, dim),
#                 nn.Dropout(dropout)
#             )
#             for _ in range(num_layers)
#         ])
        
#         # 출력 층
#         self.output_norm = nn.LayerNorm(dim)
#         self.output_layer = nn.Linear(dim, dim)
        
#         # 프로젝션 레이어
#         self.q_proj = nn.Linear(dim, dim)  # 쿼리 프로젝션
#         self.k_proj = nn.Linear(dim, dim)  # 키 프로젝션
#         self.v_proj = nn.Linear(dim, dim)  # 값 프로젝션
        
#     def forward(
#         self, 
#         x: torch.Tensor,
#         memory_state: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len, _ = x.shape
        
#         # 입력 임베딩
#         x = self.embedding(x)
        
#         # 영구 메모리 가져와서 입력에 결합
#         persistent_mem = self.persistent_memory(batch_size)
#         x_with_persistent = torch.cat([persistent_mem, x], dim=1)
        
#         # 프로젝션
#         query = self.q_proj(x_with_persistent)
#         key = self.k_proj(x_with_persistent)
#         value = self.v_proj(x_with_persistent)
        
#         # 1. 장기 메모리 레이어 (첫 번째 레이어로 처리)
#         memory_output = self.long_term_memory(query, key, value, update_memory=True)
        
#         # 메모리 상태 업데이트
#         memory_state = self.long_term_memory.memory_network[-1].weight.clone()
        
#         # 메모리 출력 정규화
#         memory_output = self.norm_layers[0](memory_output)
        
#         # 2. 어텐션 레이어 + FFN으로 순차적 처리
#         x = memory_output
#         for i in range(self.num_layers):  # 수정: self.num_layers 사용
#             # 어텐션 레이어
#             residual = x
#             x = self.norm_layers[i+1](x)
#             x = self.attention_layers[i](x)
#             x = residual + x
            
#             # FFN 레이어
#             residual = x
#             x = self.norm_layers[i+1](x)
#             x = self.ffn_layers[i](x)
#             x = residual + x
        
#         # 영구 메모리 제외한 부분만 추출
#         output = x[:, persistent_mem.size(1):]
        
#         # 출력 층
#         output = self.output_layer(self.output_norm(output))
        
#         return output, memory_state

# # 모델 사용 예시
# def example_usage():
#     # 모델 초기화
#     model = TitansMAC(
#         dim=512,
#         num_layers=4,
#         num_heads=8,
#         memory_depth=2,
#         window_size=256,
#         segment_size=128,
#         persistent_memories=16,
#         dropout=0.1
#     )
    
#     # 입력 데이터 (배치 크기 2, 시퀀스 길이 512, 차원 512)
#     x = torch.randn(2, 512, 512)
    
#     # 모델 추론
#     output, memory_state = model(x)
    
#     print(f"입력 크기: {x.shape}")
#     print(f"출력 크기: {output.shape}")
#     print(f"메모리 상태 크기: {memory_state.shape}")
    
#     return output, memory_state

# if __name__ == "__main__":
#     example_usage()