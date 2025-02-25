
import torch
from Titans import TitansMAC  # 여기를 변경했습니다

# 모델 초기화
model = TitansMAC(
    dim=512,                # 모델 차원
    num_layers=4,           # 레이어 수
    num_heads=8,            # 어텐션 헤드 수
    memory_depth=2,         # 메모리 깊이
    window_size=256,        # 어텐션 윈도우 크기
    segment_size=128,       # 세그먼트 크기
    persistent_memories=16, # 영구 메모리 개수
    dropout=0.1             # 드롭아웃 비율
)

# 입력 데이터 (예: 토큰 임베딩)
x = torch.randn(2, 512, 512)  # (배치, 시퀀스 길이, 차원)

# 모델 추론
output, memory_state = model(x)

print(f"입력 크기: {x.shape}")
print(f"출력 크기: {output.shape}")
print(f"메모리 상태 크기: {memory_state.shape}")