from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
import numpy as np
import re
#from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from . import loadModel

def test(request):
    return HttpResponse("Success")

@api_view(['POST'])
def detect(request):
    if request.method != 'POST':
        return HttpResponse("Bad Request", status=400)

    inputSentence = request.data['review']

    # X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')
    #
    # loaded_model = load_model('lstm_model.h5')
    # predictions = loaded_model.predict(X_test)
    # y_pred = (predictions > 0.5)

    # a = accuracy_score(y_test, y_pred)
    # p = precision_score(y_test, y_pred)
    # r = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    score = loadModel.confusionMatrix()
    responseBody = {}
    responseBody['acc'] = score[0]
    responseBody['precision'] = score[1]
    responseBody['recall'] = score[2]
    responseBody['f1'] = score[3]

    # okt = Okt()
    # stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    # tokenizer = Tokenizer()
    # new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', inputSentence)
    # new_sentence = okt.morphs(new_sentence, stem=True)
    # new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    # encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    # pad_new = pad_sequences(encoded, maxlen=30)  # 패딩
    # score = float(loaded_model.predict(pad_new))  # 예측
    # if (score > 0.5):
    #     responseBody['result'] = score * 100
    # else:
    #     responseBody['result'] = (1 - score) * 100
    return JsonResponse(responseBody, status=200)

