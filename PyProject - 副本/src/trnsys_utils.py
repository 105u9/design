# -*- coding: utf-8 -*-
import socket
import struct
import time
import numpy as np

class TRNSYSBridge:
    """
    Bridge for TRNSYS Type 169 communication (Socket-based).
    This allows Python to act as a controller for TRNSYS Type 56 building models.
    """
    def __init__(self, host='127.0.0.1', port=5005):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        
    def connect(self, timeout=10):
        print(f"Connecting to TRNSYS (Type 169) at {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        try:
            self.sock.connect((self.host, self.port))
            self.connected = True
            print("Connected to TRNSYS successfully!")
        except Exception as e:
            print(f"Failed to connect to TRNSYS: {e}")
            self.connected = False
            
    def receive_inputs(self, num_inputs=5):
        """Receive sensor data from TRNSYS (Building State)"""
        if not self.connected:
            return None
        try:
            # Type 169 usually sends data as a series of doubles (8 bytes each)
            data = self.sock.recv(num_inputs * 8)
            if not data:
                return None
            inputs = struct.unpack(f"{num_inputs}d", data)
            return inputs
        except Exception as e:
            print(f"Error receiving from TRNSYS: {e}")
            return None
            
    def send_outputs(self, outputs):
        """Send control commands to TRNSYS (Setpoints)"""
        if not self.connected:
            return False
        try:
            # Send as doubles
            data = struct.pack(f"{len(outputs)}d", *outputs)
            self.sock.sendall(data)
            return True
        except Exception as e:
            print(f"Error sending to TRNSYS: {e}")
            return False
            
    def close(self):
        if self.sock:
            self.sock.close()
            self.connected = False
            print("TRNSYS connection closed.")

def simulate_trnsys_loop(model, scaler):
    """Example of a closed-loop control with TRNSYS"""
    bridge = TRNSYSBridge()
    bridge.connect()
    
    if not bridge.connected:
        print("Note: TRNSYS must be running in 'Server' mode to connect.")
        return

    print("Starting Closed-Loop Control with TRNSYS...")
    try:
        while True:
            # 1. Perception: Receive current state (T_in, T_out, RH, CO2, Power)
            state = bridge.receive_inputs(5)
            if state is None:
                break
            
            # 2. Prediction: Use AI model to forecast
            # (In a real scenario, we'd build a sequence from the last N states)
            # For demo, we just show the logic
            # prediction = model(prepare_sequence(state))
            
            # 3. Optimization: Run MOPSO for current state
            # target_load = prediction[0]
            # setpoint = mopso.solve(target_load)
            
            # 4. Control: Send setpoint back to TRNSYS
            setpoint = 23.5 # Example optimized setpoint
            bridge.send_outputs([setpoint])
            
            print(f"TRNSYS State: T={state[0]:.2f}C, CO2={state[3]:.0f}ppm | Control: SP={setpoint:.1f}C")
            time.sleep(1) # Simulation step
            
    except KeyboardInterrupt:
        print("Simulation stopped.")
    finally:
        bridge.close()

if __name__ == "__main__":
    # Test without a real TRNSYS server running (will fail, as expected)
    simulate_trnsys_loop(None, None)
