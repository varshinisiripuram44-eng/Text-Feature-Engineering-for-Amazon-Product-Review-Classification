from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

from utils.predictor import predict_sentiment

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):
    context = {}

    if request.method == "POST":
        text = request.POST.get("text")

        # 🔮 Run prediction
        result, confidence = predict_sentiment(text)

        # 💾 Save to DB
        Prediction.objects.create(
            user=request.user,
            input_data={"text": text},
            predicted_class=result,
            confidence=confidence
        )

        context["result"] = result
        context["confidence"] = round(confidence, 4)

    return render(request, 'dashboard/predict.html', context)

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
