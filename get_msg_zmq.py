# consumer.py
import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "gpu_channel")

print("En attente de messages...")
while True:
    message = socket.recv_string()
    topic, payload = message.split(" ", 1)
    data = json.loads(payload)
    print("Reçu :", data)