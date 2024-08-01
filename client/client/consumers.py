import json
import logging
from channels.generic.websocket import WebsocketConsumer

logger = logging.getLogger(__name__)

class ImageConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        logger.info('WebSocket connection established.')

    def disconnect(self, close_code):
        logger.info('WebSocket connection closed. Code: %s', close_code)

    def receive(self, text_data):
        logger.info('Received data: %s', text_data[:100])  # Log the first 100 characters of the received data
        image_data = text_data
        self.send(text_data=json.dumps({'image': image_data}))
