
import numpy as np
import re
import torch
import torch.nn as nn
import joblib

from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading models...")

tfidf = joblib.load("tfidf_vectorizer.pkl")
extra_model = joblib.load("extra_trees.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear((input_dim // 2) * 64, 128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x


# Load CNN
input_dim = 5768
cnn_model = CNNFeatureExtractor(input_dim).to(device)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
cnn_model.eval()

print(" Models loaded successfully\n")


stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


user_input = "Very nice tea, like the caffeine free! I use this tea quite a bit.  It's good hot and cold, makes great iced green tea, and you don't have to worry about the caffeine."

cleaned = clean_text(user_input)


# TF-IDF
tfidf_feat = tfidf.transform([cleaned]).toarray()

# BERT
bert_feat = get_bert_embedding([cleaned])

# Combine
combined = np.hstack((tfidf_feat, bert_feat))


combined_tensor = torch.tensor(combined, dtype=torch.float32).to(device)

with torch.no_grad():
    cnn_features = cnn_model(combined_tensor).cpu().numpy()

extra_prob = extra_model.predict_proba(cnn_features)
xgb_prob = xgb_model.predict_proba(cnn_features)

final_prob = (extra_prob + xgb_prob) / 2
prediction = np.argmax(final_prob, axis=1)[0]

confidence = final_prob[0][prediction]

if prediction == 1:
    sentiment = "Positive 😊"
else:
    sentiment = "Negative 😡"

print("\n=========================")
print("Prediction:", sentiment)
print(f"Confidence: {confidence:.4f}")
print("=========================")