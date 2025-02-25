import torch
from Titans import TitansMAC, TitansMAG, TitansMAL

def test_models():
    # 공통 파라미터
    params = {
        "dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "memory_depth": 2,
        "window_size": 256,
        "persistent_memories": 16,
        "dropout": 0.1
    }
    
    # 입력 데이터 (배치 크기 2, 시퀀스 길이 512, 차원 512)
    x = torch.randn(2, 512, 512)
    
    # MAC 모델 테스트
    print("=" * 50)
    print("테스트: Memory as a Context (MAC)")
    print("=" * 50)
    mac_model = TitansMAC(**params, segment_size=128)
    mac_output, mac_memory = mac_model(x)
    print(f"입력 크기: {x.shape}")
    print(f"출력 크기: {mac_output.shape}")
    print(f"메모리 상태 크기: {mac_memory.shape}")
    
    # MAG 모델 테스트
    print("\n" + "=" * 50)
    print("테스트: Memory as a Gate (MAG)")
    print("=" * 50)
    mag_model = TitansMAG(**params)
    mag_output, mag_memory = mag_model(x)
    print(f"입력 크기: {x.shape}")
    print(f"출력 크기: {mag_output.shape}")
    print(f"메모리 상태 크기: {mag_memory.shape}")
    
    # MAL 모델 테스트
    print("\n" + "=" * 50)
    print("테스트: Memory as a Layer (MAL)")
    print("=" * 50)
    mal_model = TitansMAL(**params)
    mal_output, mal_memory = mal_model(x)
    print(f"입력 크기: {x.shape}")
    print(f"출력 크기: {mal_output.shape}")
    print(f"메모리 상태 크기: {mal_memory.shape}")
    
    # 출력 비교
    print("\n" + "=" * 50)
    print("모델 출력 비교 (L2 norm)")
    print("=" * 50)
    print(f"MAC vs MAG: {torch.norm(mac_output - mag_output):.4f}")
    print(f"MAC vs MAL: {torch.norm(mac_output - mal_output):.4f}")
    print(f"MAG vs MAL: {torch.norm(mag_output - mal_output):.4f}")

if __name__ == "__main__":
    test_models()