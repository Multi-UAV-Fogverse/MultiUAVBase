import imagezmq
import cv2
import asyncio
import websockets
import base64
import logging

logging.basicConfig(level=logging.DEBUG)

async def send_image():
    image_hub = imagezmq.ImageHub(open_port='tcp://0.0.0.0:5001')

    async with websockets.connect('ws://base_client:8000/ws/image/') as websocket:
        while True:
            rpi_name, image = image_hub.recv_image()
            _, buffer = cv2.imencode('.jpg', image)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            logging.debug('Sending image data of length: %d', len(jpg_as_text))
            await websocket.send(jpg_as_text)

asyncio.get_event_loop().run_until_complete(send_image())
