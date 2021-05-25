from django.shortcuts import render
from django.shortcuts import redirect
from .forms import UploadFileForm
import pandas as pd
from hooks.Predictor import Predict

def home(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            f = request.FILES['file']
            with open('dashboard/data/input.csv', 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
            
            reviews_to_predict = pd.read_csv('dashboard/data/input.csv', sep='\t')
            reviews_to_predict = pd.DataFrame(reviews_to_predict)

            output = Predict(reviews_to_predict)
            output.to_csv('dashboard/data/output.txt', sep='\t', index=False)
            
            return redirect('dashboard')
    else:
        form = UploadFileForm()
    return render(request, 'home.html', {'form': form})
