import streamlit as st
import tempfile
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from predictor import ProblemPredictor, Configuration, LMForMultiLabelClassification
from datetime import datetime
from mathpix.mathpix import MathPix
from transformers import AutoModel, AutoTokenizer
from collections import OrderedDict
from PIL import Image
import io
import time

# 필요한 인스턴스를 생성합니다. 
# 이때, model, tokenizer, device, threshold, Config는 사용자가 정의한 값을 사용해야 합니다.
config = Configuration(model_name="klue/bert-base", max_length=128, batch_size=8, padding=True, truncation=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.5

LM = AutoModel.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Define the model
model = LMForMultiLabelClassification(LM, num_labels=5)
model_load_path = './model_last.pt'

# Load the model
checkpoint = torch.load(model_load_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
threshold = 0.5

predictor = ProblemPredictor(model, tokenizer, device, threshold, config)

# 페이지 설정
st.set_page_config(
    page_title="수학문제 인지요소 분류기",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 정보
with st.sidebar:
    st.markdown("## 📊 프로젝트 정보")
    st.markdown("**개발팀**: 이효준, 최동민")
    st.markdown("**모델**: KLUE/BERT-base")
    st.markdown("**분류 카테고리**: 6개")
    st.markdown("**임계값**: 0.5")
    
    st.markdown("---")
    st.markdown("## 🎯 분류 카테고리")
    categories = [
        "1️⃣ 유한소수와 무한소수의 이해",
        "2️⃣ 순환소수의 이해", 
        "3️⃣ 유한소수로 표현되는 분수 구별하기",
        "4️⃣ 분수의 순환소수 표현",
        "5️⃣ 순환소수의 분수 표현",
        "6️⃣ 유리수와 순환소수의 관계 이해"
    ]
    for cat in categories:
        st.markdown(f"- {cat}")

# 메인 헤더
st.markdown("""
<div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
    <h1 style='color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.3rem;'>🧮 수학문제 인지요소 분류기</h1>
    <h3 style='color: #666; font-weight: 300; margin-bottom: 1rem;'>AI 기반 수학 문제 이미지 분석 및 자동 분류 시스템</h3>
    <div style='background: linear-gradient(90deg, #1f77b4, #ff7f0e); height: 3px; border-radius: 2px; margin: 0.5rem auto; width: 150px;'></div>
</div>
""", unsafe_allow_html=True)

# 시스템 정보 카드
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 0.7rem; border-radius: 8px; border-left: 3px solid #1f77b4;'>
        <h5 style='color: #1f77b4; margin: 0; font-size: 0.9rem;'>🔍 OCR 엔진</h5>
        <p style='margin: 0.3rem 0 0 0; color: #666; font-size: 0.8rem;'>MathPix API</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #fff8f0; padding: 0.7rem; border-radius: 8px; border-left: 3px solid #ff7f0e;'>
        <h5 style='color: #ff7f0e; margin: 0; font-size: 0.9rem;'>🤖 AI 모델</h5>
        <p style='margin: 0.3rem 0 0 0; color: #666; font-size: 0.8rem;'>KLUE/BERT-base</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background-color: #f0fff0; padding: 0.7rem; border-radius: 8px; border-left: 3px solid #2ca02c;'>
        <h5 style='color: #2ca02c; margin: 0; font-size: 0.9rem;'>⚡ 후처리</h5>
        <p style='margin: 0.3rem 0 0 0; color: #666; font-size: 0.8rem;'>규칙 기반 엔진</p>
    </div>
    """, unsafe_allow_html=True)

# 파일 업로드 섹션
st.markdown("### 📤 이미지 업로드")
st.markdown("수학 문제가 포함된 이미지를 업로드하세요.")

mathpix = MathPix(app_id=st.secrets["app_id"], app_key=st.secrets["app_key"])

uploaded_file = st.file_uploader(
    "이미지 파일을 선택하세요", 
    type=['png', 'jpg', 'jpeg'],
    help="Sample 폴더의 예시 이미지들을 사용해보세요!"
)

if uploaded_file is not None:
    # 파일 포인터를 처음으로 되돌리기
    uploaded_file.seek(0)
    
    with st.spinner("이미지를 분석하고 있습니다..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            ocr = mathpix.process_image(temp_path)
            new_data = [ocr.latex]
            result = predictor.predict_problem(new_data)
            
        except Exception as e:
            st.error(f"❌ 처리 중 오류가 발생했습니다: {str(e)}")
            st.error("💡 **해결 방법**: 다른 이미지를 시도하거나 이미지 형식(PNG, JPG)을 확인해주세요.")
            st.stop()
    
    # 이미지와 결과를 좌우로 배치
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 업로드된 이미지")
        image = Image.open(uploaded_file)
        st.image(image, caption=f"파일명: {uploaded_file.name}", use_column_width=True)
        
        # OCR 결과
        with st.expander("🔍 OCR 추출 결과", expanded=False):
            if ocr.latex:
                st.markdown("**추출된 LaTeX:**")
                st.code(ocr.latex, language="latex")
            else:
                st.warning("텍스트를 추출할 수 없었습니다.")
    
    with col2:
        st.markdown("### 🎯 예측된 수학 개념")
        
        if result and result != "":
            predicted_concepts = result.split(", ")
            
            # 결과를 카드 형태로 표시
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                <h5 style='color: white; text-align: center; margin-bottom: 0.5rem; font-size: 1rem;'>
                    🎉 분석된 수학 개념들
                </h5>
            """, unsafe_allow_html=True)
            
            # 개념별 카드 표시
            for i, concept in enumerate(predicted_concepts):
                concept = concept.strip()
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']
                color = colors[i % len(colors)]
                
                st.markdown(f"""
                <div style='background-color: white; margin: 0.3rem 0; padding: 0.6rem; 
                            border-radius: 6px; border-left: 3px solid {color};
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <p style='color: {color}; margin: 0; font-size: 0.9rem; font-weight: 500;'>
                        ✓ {concept}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
                
        else:
            st.warning("⚠️ 해당 이미지에서 분류 가능한 수학 개념을 찾을 수 없습니다.")
    
    # 정리
    os.unlink(temp_path)

else:
    # 사용법 안내
    st.markdown("## 📋 사용 방법")
    
    steps = [
        "📤 위의 파일 업로더를 통해 수학 문제 이미지를 업로드하세요",
        "🔄 시스템이 자동으로 OCR → AI 분류 → 결과 출력을 진행합니다", 
        "📊 예측된 수학 개념들을 확인하세요",
        "💡 Sample 폴더의 예시 이미지들로 테스트해보세요!"
    ]
    
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")
    
    st.markdown("---")
    st.markdown("### 🚀 시작하기")
    st.info("💡 **팁**: Sample 폴더에 있는 예시 이미지들을 사용하여 시스템을 테스트해보세요!")