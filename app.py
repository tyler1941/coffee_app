import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import openai  # 📌 OpenAI API 추가

# ✅ OpenAI API 키 설정 (🔹 여기에 본인의 API 키를 입력해야 함)
openai.api_key = "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"

# ✅ 모델 클래스 정의
class NailDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(NailDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ 학습된 모델 불러오기
num_classes = 6
model = NailDiseaseCNN(num_classes)
model.load_state_dict(torch.load("nail_disease_model.pth", map_location=torch.device("cpu")))
model.eval()

# ✅ 클래스 이름 (한글 변환)
class_names = [
    "말단 흑색점 흑색종",
    "건강한 손톱",
    "조갑백선증",
    "청색 손가락",
    "곤봉지",
    "손톱 오목증"
]

# ✅ 이미지 전처리 함수
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# ✅ OpenAI API를 이용한 AI 의사 진단서 생성 함수 (최신 방식)
def generate_medical_report(disease_name):
    prompt = f"""
    당신은 피부과 전문의입니다. 아래의 질병에 대해 진단서를 작성해주세요.

    - 질병명: {disease_name}
    - 질병의 주요 증상
    - 치료 방법
    - 병원 방문이 필요한 경우
    - 예방법
    - 추가적인 주의사항
    """

    response = openai.chat.completions.create(
        model="gpt-4",  # 최신 GPT-4 모델 사용
        messages=[
            {"role": "system", "content": "당신은 피부과 전문의입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content

# ✅ Streamlit UI 구성
st.title("🩺 손톱 질병 진단 AI")
st.write("손톱 사진을 업로드하면 AI 의사가 질병을 예측하고 진단서를 작성합니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 업로드된 이미지", use_column_width=True)

    # ✅ 이미지 변환 후 예측 수행
    input_tensor = transform_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]  # 확률값 변환
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = class_names[predicted_index]
        predicted_probability = probabilities[predicted_index].item() * 100  # 확률을 %로 변환

    # ✅ 결과 출력
    st.write(f"## 🔍 예측 결과: **{predicted_label}**")
    st.write(f"📊 예측 확률: **{predicted_probability:.2f}%**")

    # ✅ OpenAI API를 이용해 AI 의사의 진단서 생성
    with st.spinner("🩺 AI 의사가 진단서를 작성 중입니다..."):
        medical_report = generate_medical_report(predicted_label)

    # ✅ 진단서 출력
    st.markdown("## 📋 AI 의사의 진단서")
    st.markdown(medical_report)

    # ✅ 상위 3개 예측 결과 표시
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    st.write("🔹 **상위 3개 예측 결과**")
    for i in range(3):
        st.write(f"{class_names[top3_idx[i].item()]}: {top3_prob[i].item() * 100:.2f}%")
