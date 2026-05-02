# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import json
import time
import threading

class MQTTClient:
    def __init__(self, broker="broker.emqx.io", port=1883, topic="hvac/sensors"):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.latest_data = {}
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT Broker: {self.broker}")
            self.client.subscribe(self.topic)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.latest_data = payload
            # print(f"Received from {msg.topic}: {payload}")
        except Exception as e:
            print(f"Error parsing MQTT message: {e}")

    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port, 60)
        # Run in a background thread
        self.thread = threading.Thread(target=self.client.loop_forever)
        self.thread.daemon = True
        self.thread.start()

    def publish(self, data):
        self.client.publish(self.topic, json.dumps(data))

if __name__ == "__main__":
    # Test Publisher
    pub = MQTTClient()
    pub.client.connect("broker.emqx.io", 1883, 60)
    print("Simulating IoT Sensor Publishing...")
    try:
        while True:
            test_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "indoor_temp": 24.5,
                "indoor_humidity": 45.0,
                "indoor_co2": 650,
                "power_usage": 12.5
            }
            pub.publish(test_data)
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopped.")
