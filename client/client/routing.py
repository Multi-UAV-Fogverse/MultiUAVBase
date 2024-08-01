from django.urls import re_path
from client import consumers

websocket_urlpatterns = [
    re_path(r'ws/image/$', consumers.ImageConsumer.as_asgi()),
]
