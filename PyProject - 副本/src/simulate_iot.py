# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mqtt_utils import MQTTClient
from preprocessing import load_building_data, clean_and_impute

def simulate_iot_publisher(building_id="Panther_office_Karla", site_id="Panther"):
    print(f"=== Starting IoT Sensor Simulation (MQTT) for {building_id} ===")
    
    # Load and clean data
    print("Loading building data for simulation...")
    df = load_building_data(building_id, site_id)
    df = clean_and_impute(df, method='mean')
    
    # Use a free public MQTT broker (EMQX)
    broker = "broker.emqx.io"
    port = 1883
    topic = "hvac/sensors/data"
    
    client = MQTTClient(broker=broker, port=port, topic=topic)
    client.client.connect(broker, port, 60)
    
    print(f"Publishing to {broker}:{port} on topic {topic}...")
    print("Press Ctrl+C to stop.")
    
    # Simulate data flow (1 sample every 2 seconds)
    try:
        for idx, row in df.iterrows():
            data = {
                "timestamp": str(row['timestamp']),
                "air_temp": float(row['airTemperature']),
                "power_usage": float(row['power_usage']),
                "indoor_temp": float(row['indoor_temp']),
                "indoor_humidity": float(row['indoor_humidity']),
                "indoor_co2": float(row['indoor_co2'])
            }
            client.publish(data)
            print(f"[{data['timestamp']}] Published: Temp={data['indoor_temp']:.2f}C, CO2={data['indoor_co2']:.0f}ppm, Power={data['power_usage']:.2f}kW")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    simulate_iot_publisher()
