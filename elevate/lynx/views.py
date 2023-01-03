from django.shortcuts import render

# - Homepage

def home(request):

    return render(request, 'lynx/index.html')