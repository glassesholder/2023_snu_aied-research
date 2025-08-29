import nltk
import re
import torch
import torch.nn as nn
import torch.optim as optim

# nltk는 postprocess를 위해 import합니다 - postprocess_text 함수 참고
nltk.download('punkt')
nltk.download('stopwords')

class Configuration():
    def __init__(self, model_name="klue/bert-base", max_length=256, batch_size=8, padding=True, truncation=True):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.padding = padding
        self.truncation = truncation

class LMForMultiLabelClassification(nn.Module):
    def __init__(self, LM, num_labels):
        super(LMForMultiLabelClassification, self).__init__()
        self.LM = LM
        self.hidden_size = self.LM.config.hidden_size
        self.classificationHead = nn.Linear(self.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.LM(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.classificationHead(outputs.last_hidden_state[:, 0, :])
        probs = self.sigmoid(logits)
        return probs

class ProblemPredictor:
    def __init__(self, model, tokenizer, device, threshold, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.config = config

    def preprocess_question(self, text):
        if text is None:
            text = ''
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\\', '', text)
        text = re.sub(r'[^가-힣 ]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'(개\s?){5}', '', text)
        return text

    def find_closest_to_threshold(self, predictions):
        diff_from_threshold = [abs(pred - self.threshold) for pred in predictions]
        closest_index = diff_from_threshold.index(min(diff_from_threshold))
        return closest_index

    def predict_problem(self, new_data):
        
        print(self.model)
        
        preprocessed_new_data = [self.preprocess_question(text) for text in new_data]
        new_inputs = self.tokenizer(preprocessed_new_data, padding=self.config.padding, truncation=self.config.truncation, return_tensors="pt", max_length=self.config.max_length)
        new_outputs = self.model(input_ids=new_inputs.input_ids.to(self.device), attention_mask=new_inputs.attention_mask.to(self.device), token_type_ids=new_inputs.token_type_ids.to(self.device))
        print(new_outputs)
        print(self.threshold)
        new_predictions = (new_outputs > self.threshold).int().cpu().numpy().tolist()
        new_predictions[0].insert(0, 1)

        # 추가 규칙1: 6번째 값이 1이 나왔을 경우 적어도 3, 4, 5번째 값 중 하나는 1이 되어야 함
        if new_predictions[0][5] == 1 and all(val == 0 for val in new_predictions[0][2:5]):
            # threshold에 가장 근접한 값을 1로 설정
            closest_index = self.find_closest_to_threshold(new_outputs[0][2:5].tolist())
            new_predictions[0][closest_index + 2] = 1  # closest_index + 2를 사용하여 원래 인덱스로 변환

        # 추가 규칙2: 3, 4, 5번째 값 중 하나가 1이 나왔을 경우 반드시 2번째 값이 1이어야 함
        if any(new_predictions[0][2:5]):
            new_predictions[0][1] = 1
        
        labels = [
            '1.유한소수와 무한소수의 이해',
            '2.순환소수의 이해',
            '3.유한소수로 표현되는 분수 구별하기',
            '4.분수의 순환소수 표현',
            '5.순환소수의 분수 표현',
            '6.유리수와 순환소수의 관계 이해'
        ]

        found_labels = [label for i, label in enumerate(labels) if new_predictions[0][i] == 1]

        return ", ".join(found_labels)
