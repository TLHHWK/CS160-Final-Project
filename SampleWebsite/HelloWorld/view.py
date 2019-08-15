import os
import simplejson as json
from django.http import HttpResponse


def hello(request):
 # print(request)
 # print(request.GET)
 Index=request.GET.get('data')
 type = request.GET.get('type')
 name = request.GET.get('name')
 print(Index)
 print(type)
 print(name)
 os.system("python D:\BerkeleyFinal\Style-Transfer\code.py "+Index+" "+type+" "+name)
 return HttpResponse("Hello world !")