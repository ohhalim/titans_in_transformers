import torch
from titans_model import EnhancedTitansTransformer

def main():
    # 모델 파라미터 설정
    config = {
        'd_model': 512,            # dim -> d_model로 변경
        'nhead': 8,               # num_heads -> nhead로 변경
        'num_layers': 6,
        'vocab_size': 30522,      # 추가 필요
        'persistent_mem_len': 16,
        'segment_length': 128
    }

    # 모델 초기화 (TitansModel -> EnhancedTitansTransformer로 변경)
    model = EnhancedTitansTransformer(**config)

    # GPU 사용 가능시 GPU로 모델 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # 샘플 입력 데이터 생성 (입력 형식 변경)
    batch_size = 2
    seq_length = 128
    
    # 토큰 인덱스로 입력 생성 (랜덤 텐서 대신)
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_length)).to(device)

    # 모델을 평가 모드로 설정
    model.eval()

    # 순전파 실행 (반환값이 logits 하나)
    with torch.no_grad():
        output = model(x)

    # 결과 출력
    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)

    # 출력의 통계값 확인
    print("\nOutput statistics:")
    print("Mean:", output.mean().item())
    print("Std:", output.std().item())
    print("Min:", output.min().item())
    print("Max:", output.max().item())

if __name__ == "__main__":
    main()