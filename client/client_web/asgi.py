import os
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import client.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'client_web.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            client.routing.websocket_urlpatterns
        )
    ),
})
