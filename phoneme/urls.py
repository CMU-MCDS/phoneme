from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^upload_train$',views.upload_train,name='upload_train'),
    url(r'^upload_transcribe',views.upload_transcribe,name='upload_transcribe'),
    url(r'^train$',views.train,name='train'),
    url(r'^transcribe$',views.transcribe,name='transcribe'),
]