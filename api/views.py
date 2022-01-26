from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from . import loadModel

@api_view(['POST'])
def detect(request):
    if request.method != 'POST':
        return HttpResponse("Bad Request", status=400)

    inputSentence = request.data['review']
    score = loadModel.confusionMatrix(inputSentence)
    responseBody = {}
    responseBody['input_score'] = 'None' if score[0] == 1 else 'Hate'
    responseBody['acc'] = score[1]
    responseBody['precision'] = score[2]
    responseBody['recall'] = score[3]
    responseBody['f1'] = score[4]
    return JsonResponse(responseBody, status=200)

