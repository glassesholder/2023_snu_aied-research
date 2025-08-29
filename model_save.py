PATH = './model_last.pt'

# 모델을 저장할 경로를 지정합니다.
model_save_path = PATH

# 모델의 상태 사전 및 다른 중요 매개변수를 저장합니다.
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,  # 현재 학습 에포크
    'loss': running_loss  # 학습 손실
}

# 저장합니다.
torch.save(checkpoint, model_save_path)

model_load_path = './model_last.pt'

# 미리 정의한 모델 아키텍처를 생성합니다.
loaded_model = LMForMultiLabelClassification(LM, num_labels=5)

# 저장된 모델의 상태 사전 및 다른 중요 매개변수를 불러옵니다.
checkpoint = torch.load(model_load_path)

# 모델의 가중치를 불러옵니다.
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# 옵티마이저의 상태 및 다른 정보를 불러옵니다.
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 이전 학습 에포크 및 손실을 불러옵니다.
epoch = checkpoint['epoch']
loss = checkpoint['loss']
