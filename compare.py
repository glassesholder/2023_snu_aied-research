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

#페이지의 메인 이름
st.title("학생연구 장기 프로젝트 5조 결과물")

#가로 줄
st.divider()

#헤더 
st.header("Mathpix API 활용")

mathpix = MathPix(app_id=st.secrets["app_id"], app_key=st.secrets["app_key"])

uploaded_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Pass the file path to the Mathpix API
    ocr = mathpix.process_image(temp_path)

    new_data = [ocr.latex]  # new_data는 실제 사용할 데이터로 바꿔야 합니다.
    result = predictor.predict_problem(new_data)
    
    st.write(result) # 결과를 출력합니다.