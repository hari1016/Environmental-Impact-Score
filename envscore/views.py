from django.shortcuts import render

# Create your views here.
import random
from django.shortcuts import render

from .ml_model import predict_score



def evaluate_product(request):
    score = -1
    if request.method == 'POST':
        product_name = request.POST.get('productName')

        score = predict_score(product_name)
        return render(request, 'evaluate.html', {'product_name': product_name, 'score': score})
    return render(request, 'evaluate.html', {'score': score})
