import numpy as np
import pickle
import re
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def confusionMatrix(inputSentence):
    X_test = np.load('model/X_test.npy')
    y_test = np.load('model/y_test.npy')

    #loaded_model = load_model('lstm_model.h5', compile=False)
    loaded_model = pickle.load(open("model/lstm_model.pkl", 'rb'))
    predictions = loaded_model.predict(X_test)
    y_pred = (predictions > 0.5)

    a = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    #print(classification_report(y_test, y_pred, target_names=['hate', 'none']))
    score = []
    inputResult = sentencePredict(inputSentence, loaded_model)
    score.append(inputResult)
    score.append(a)
    score.append(p)
    score.append(r)
    score.append(f1)
    return score

def sentencePredict(inputSentence, model):
    okt = Okt()
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    tokenizer = Tokenizer()
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', inputSentence)
    new_sentence = okt.morphs(new_sentence, stem=True)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=30)
    pred = float(model.predict(pad_new))  # 예측
    print("pred : ", pred)
    if (pred > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(pred * 100))
        return 1
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - pred) * 100))
        return 0

if __name__ == '__main__':
    cm = confusionMatrix()