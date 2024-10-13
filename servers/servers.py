import asyncio
import queue
import struct
import numpy as np
import cv2
from ultralytics import YOLO
import os
import signal
import subprocess
import sys
import time
import zmq
import json
import matplotlib.pyplot as plt
from kalmanfilter import KF
from kalmanfilter import EKF
from kalman_evaluation import KalmanFilterEvaluator
import datetime

# Determine the base path for the script
if getattr(sys, "frozen", False):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(
        os.path.abspath(__file__)))
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Set the path for the YOLO model
model_path = os.path.join(base_path, "best_2nd_run.pt")

# Initialize global variables for evaluators
worker_evaluator = None
excavator_evaluator = None


def initialize_evaluators():
    global worker_evaluator, excavator_evaluator
    worker_evaluator = KalmanFilterEvaluator("Worker KF")
    excavator_evaluator = KalmanFilterEvaluator("Excavator EKF")

# Initialize lists for storing measurement and estimation data
measurement_x_worker, measurement_z_worker = [], []
estimated_x_kf_worker, estimated_z_kf_worker = [], []
predicted_x_worker, predicted_z_worker = [], []
true_x_worker, true_z_worker = [], []
true_vx_worker, true_vz_worker = [], []

measurement_x_excavator, measurement_z_excavator = [], []
estimated_x_ekf_excavator, estimated_z_ekf_excavator = [], []
predicted_x_excavator, predicted_z_excavator = [], []
true_x_excavator, true_z_excavator = [], []
true_vx_excavator, true_vz_excavator = [], []

measured_vx, measured_vz = [], []
estimated_vx, estimated_vz = [], []
predicted_vx, predicted_vz = [], []

# Set server configuration
SERVER_HOST = "127.0.0.1"
PORT1 = 9999
PORT2 = 10000

# Initialize YOLO model
model = YOLO(model_path)

# Create an event for graceful shutdown
shutdown_event = asyncio.Event()

# Initialize server variables
frame_server = None
measurement_server = None

# Record start time
start_time = time.time()

# Create queues for frames and measurements
frame_queue = queue.Queue(maxsize=30)
measurement_queue = queue.Queue(maxsize=30)

# Initialize variables for frame processing
last_processed_frame_number = 0
number_frames_received = 0

# Set up ZeroMQ for sending results to Result Server
context = zmq.Context()
result_sender = context.socket(zmq.PUSH)
result_sender.connect("tcp://localhost:5555")


async def handle_frame_client(reader, writer):
    """Handle incoming frame data from clients."""
    global number_frames_received
    try:
        while not shutdown_event.is_set():
            try:
                # Read frame number
                frame_number_bytes = await asyncio.wait_for(
                    reader.readexactly(4), timeout=1.0
                )
                frame_number = struct.unpack(">I", frame_number_bytes)[0]
                # Read frame data
                frame_buffer = await asyncio.wait_for(
                    reader.readexactly(921600), timeout=1.0
                )
                number_frames_received += 1
                # Put frame data in queue
                frame_queue.put_nowait((frame_number, frame_buffer))
            except asyncio.TimeoutError:
                continue
            except asyncio.IncompleteReadError:
                print("Frame client disconnected")
                break
    except Exception as e:
        print(f"Error in handle_frame_client: {e}")
    finally:
        # Close the writer and set shutdown event
        writer.close()
        await writer.wait_closed()
        print("Frame client connection closed")
        shutdown_event.set()


async def handle_measurement_client(reader, writer):
    """Handle incoming measurement data from clients."""
    try:
        while not shutdown_event.is_set():
            try:
                # Read packet size
                packet_size_buffer = await asyncio.wait_for(
                    reader.readexactly(4), timeout=1.0
                )
                packet_size = struct.unpack(">I", packet_size_buffer)[0]
                writer.write(b"<ACK_SIZE>")
                await writer.drain()
                # Read measurement data
                measurement_data = await asyncio.wait_for(
                    reader.readexactly(packet_size), timeout=1.0
                )
                measurement_queue.put_nowait(measurement_data)
            except asyncio.TimeoutError:
                continue
            except asyncio.IncompleteReadError:
                print("Measurement client disconnected")
                break
    except Exception as e:
        print(f"Error in handle_measurement_client: {e}")
    finally:
        # Close the writer and set shutdown event
        writer.close()
        await writer.wait_closed()
        print("Measurement client connection closed")
        shutdown_event.set()


def process_frames():
    """Process frames from the frame queue."""
    global last_processed_frame_number
    window_size = 10
    frame_buffer = []

    while not shutdown_event.is_set():
        try:
            # Get frame from queue
            frame_number, frame_data = frame_queue.get(timeout=1)

            # Add frame to buffer and sort
            frame_buffer.append((frame_number, frame_data))
            frame_buffer.sort(key=lambda x: x[0])

            # Process frames in order
            while frame_buffer and frame_buffer[0][0] == last_processed_frame_number + 1:  # noqa
                current_frame_number, current_frame_data = frame_buffer.pop(0)
                process_single_frame(current_frame_number, current_frame_data)
                last_processed_frame_number = current_frame_number

            # Process oldest frame if buffer is full
            if len(frame_buffer) >= window_size:
                oldest_frame_number, oldest_frame_data = frame_buffer.pop(0)
                process_single_frame(oldest_frame_number, oldest_frame_data)
                last_processed_frame_number = oldest_frame_number
                print(f"Processed out-of-order frame: {oldest_frame_number}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in process_frames: {e}")


def process_single_frame(frame_number, frame_data):
    """Process a single frame of data."""
    try:
        start_time = time.time()

        # Preprocess the frame
        np_array = np.frombuffer(frame_data, np.uint8)
        frame = np_array.reshape((480, 640, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocess_time = time.time() - start_time
        print(f"Processing frame {frame_number}")

        # Perform inference
        inference_start = time.time()
        result = model(
            source=frame, show=False, conf=0.56, device=0, save=False
        )
        inference_time = time.time() - inference_start

        # Postprocess the results
        postprocess_start = time.time()
        detections = []
        for det in result[0].boxes:
            bbox = det.xyxy[0].tolist()
            boxes = det.xywh[0].tolist()
            conf = det.conf.item()
            label = det.cls.item()
            label_name = model.names[int(label)]
            detections.append({
                "bbox": bbox,
                "center": (boxes[0], boxes[1]),
                "label": label_name,
                "confidence": conf,
            })

        detection_result = {
            "frame_number": frame_number, "detections": detections
        }
        postprocess_time = time.time() - postprocess_start
        print(
            f"Processed frame {frame_number} with {len(detections)} detections"
        )
        # Send result to result_server
        result_sender.send_json(detection_result)
        print(
            f"Sent detection result for frame {frame_number} to Result Server"
        )

        total_time = time.time() - start_time
        print(
            f"Processed frame {frame_number} with {len(detections)} detections. "  # noqa
            f"Preprocess: {preprocess_time:.2f}s, Inference: {inference_time:.2f}s, "  # noqa
            f"Postprocess: {postprocess_time:.2f}s, Total: {total_time:.2f}s"
        )

    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")


def process_measurements():
    """Process measurements from the measurement queue."""
    global kf, ekf

    # Initialize Kalman Filters
    kf = KF(initial_x=0, initial_xv=0, initial_z=0, initial_z_v=0, dt=1 / 30)
    ekf = EKF(
        initial_x=-18.286,
        initial_z=-5.988,
        initial_vx=0.0,
        initial_vz=0.0,
        initial_theta=0.0,
        initial_omega=0.0,
        dt=1 / 30
    )

    while not shutdown_event.is_set():
        try:
            # Get measurement data from queue
            measurement_data = measurement_queue.get(timeout=1)
            json_string = measurement_data.decode('utf-8')
            measurement_dict = json.loads(json_string)
            for key in measurement_dict:
                if key == 'worker':
                    # Process worker measurements
                    measurement_worker = measurement_dict['worker'][0]
                    detected_position = measurement_worker[:2]
                    true_position = measurement_worker[2:4]
                    true_velocity = measurement_worker[4:6]
                    measured_velocity = measurement_worker[6:8]

                    # Kalman Filter prediction and update for worker
                    predicted_state, _ = kf.predict()
                    estimated_state = kf.estimate(detected_position)

                    # Store worker data
                    measurement_x_worker.append(detected_position[0])
                    measurement_z_worker.append(detected_position[1])
                    estimated_x_kf_worker.append(float(estimated_state[0]))
                    estimated_z_kf_worker.append(float(estimated_state[2]))
                    predicted_x_worker.append(float(predicted_state[0]))
                    predicted_z_worker.append(float(predicted_state[2]))
                    true_x_worker.append(true_position[0])
                    true_z_worker.append(true_position[1])
                    true_vx_worker.append(true_velocity[0])
                    true_vz_worker.append(true_velocity[1])

                elif key == 'excavator':
                    # Process excavator measurements
                    measurement_excavator = measurement_dict['excavator'][0]
                    detected_position = measurement_excavator[:2]
                    true_position = measurement_excavator[2:4]
                    true_velocity = measurement_excavator[4:6]
                    measured_velocity = measurement_excavator[6:8]

                    # Create full measurement vector for EKF
                    full_measurement = np.array([
                        detected_position[0],
                        detected_position[1],
                        measured_velocity[0],
                        measured_velocity[1],
                        0,  # Placeholder for theta (if not available)
                        0   # Placeholder for omega (if not available)
                    ])

                    # Extended Kalman Filter prediction
                    # and update for excavator
                    predicted_state = ekf.predict()
                    ekf.update(full_measurement)
                    estimated_state = ekf.get_state()

                    # Store excavator data
                    measurement_x_excavator.append(detected_position[0])
                    measurement_z_excavator.append(detected_position[1])
                    estimated_x_ekf_excavator.append(float(estimated_state[0]))
                    estimated_z_ekf_excavator.append(float(estimated_state[1]))
                    predicted_x_excavator.append(float(predicted_state[0]))
                    predicted_z_excavator.append(float(predicted_state[1]))
                    true_x_excavator.append(true_position[0])
                    true_z_excavator.append(true_position[1])
                    true_vx_excavator.append(true_velocity[0])
                    true_vz_excavator.append(true_velocity[1])
                    measured_vx.append(measured_velocity[0])
                    measured_vz.append(measured_velocity[1])
                    estimated_vx.append(float(estimated_state[2]))
                    estimated_vz.append(float(estimated_state[3]))
                    predicted_vx.append(float(predicted_state[2]))
                    predicted_vz.append(float(predicted_state[3]))

            print(f"Processed measurements: {list(measurement_dict.keys())}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in process_measurements: {e}")


async def start_server(handle_client, port):
    """Start a server to handle client connections."""
    server = await asyncio.start_server(handle_client, SERVER_HOST, port)
    addr = server.sockets[0].getsockname()
    print(f"Server listening on {addr}")

    async with server:
        try:
            await shutdown_event.wait()
        except asyncio.CancelledError:
            print("Shutdown event received.")
        finally:
            server.close()
            await server.wait_closed()
            print("Server closed.")


async def shutdown(signal, loop):
    """Handle graceful shutdown of the server."""
    print(f"Received exit signal {signal.name}...")
    shutdown_event.set()

    global frame_server, measurement_server
    if frame_server:
        frame_server.close()
    if measurement_server:
        measurement_server.close()

    if "result_sender" in globals():
        result_sender.close()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def run_result_server():
    """Run the result server as a separate process."""
    subprocess.Popen(["python", "result_server.py"])


async def main():
    """Main function to run the server."""
    global frame_server, measurement_server

    # Get the current event loop
    loop = asyncio.get_running_loop()

    # Set up signal handlers for graceful shutdown
    signals = (signal.SIGINT, signal.SIGTERM)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )

    # Start the result server in a separate process
    run_result_server()

    # Initialize Kalman filter evaluators
    initialize_evaluators()
    print("Starting frame and measurement threads")

    # Create threads for processing frames and measurements
    frame_thread = asyncio.to_thread(process_frames)
    measurement_thread = asyncio.to_thread(process_measurements)

    try:
        # Start the frame and measurement servers
        frame_server = await asyncio.start_server(
            handle_frame_client, SERVER_HOST, PORT1
        )
        measurement_server = await asyncio.start_server(
            handle_measurement_client, SERVER_HOST, PORT2
        )

        print(f"Frame server listening on {SERVER_HOST}:{PORT1}")
        print(f"Measurement server listening on {SERVER_HOST}:{PORT2}")

        # Run all servers and processing threads concurrently
        await asyncio.gather(
            frame_server.serve_forever(),
            measurement_server.serve_forever(),
            frame_thread,
            measurement_thread,
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        print("Servers were cancelled")
    finally:
        # Ensure servers are closed properly
        if frame_server:
            frame_server.close()
            await frame_server.wait_closed()
        if measurement_server:
            measurement_server.close()
            await measurement_server.wait_closed()
        print("Servers shut down gracefully.")


def plot_and_save_data():
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")  # noqa: E501
    os.makedirs(figures_dir, exist_ok=True)

    # Subdirectories for each type of Kalman filter
    worker_dir = os.path.join(figures_dir, "worker")
    excavator_dir = os.path.join(figures_dir, "excavator")

    # Ensure subdirectories exist
    os.makedirs(worker_dir, exist_ok=True)
    os.makedirs(excavator_dir, exist_ok=True)

    # Current time for unique file naming
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Plotting for worker
    plt.figure(figsize=(10, 6))
    plt.scatter(measurement_x_worker, measurement_z_worker, color="blue", label="Measurements", marker="o")
    plt.scatter(
        estimated_x_kf_worker, estimated_z_kf_worker, color="red",
        label="KF Estimations", marker="^"
    )
    plt.plot(estimated_x_kf_worker, estimated_z_kf_worker, color="red")
    plt.scatter(
        predicted_x_worker, predicted_z_worker, color="green",
        label="KF Predictions", marker="x"
    )
    plt.scatter(
        true_x_worker, true_z_worker, color="black",
        label="True Positions", marker="s"
    )
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title("KF: Measurements, Estimations, Predictions, and True Positions for Worker")  # noqa
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(worker_dir, f"{current_time}_kf_plot_worker.png"))
    plt.close()

    # Plotting for excavator positions
    plt.figure(figsize=(10, 6))
    plt.scatter(
        measurement_x_excavator, measurement_z_excavator, color="blue",
        label="Measurements", marker="o"
    )
    plt.scatter(
        estimated_x_ekf_excavator, estimated_z_ekf_excavator, color="red",
        label="EKF Estimations", marker="^"
    )
    plt.plot(estimated_x_ekf_excavator, estimated_z_ekf_excavator, color="red")
    plt.scatter(
        predicted_x_excavator, predicted_z_excavator, color="green",
        label="EKF Predictions", marker="x"
    )
    plt.scatter(
        true_x_excavator, true_z_excavator, color="black",
        label="True Positions", marker="s"
    )
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title("EKF: Measurements, Estimations, Predictions, and True Positions for Excavator")  # noqa
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(
        excavator_dir, f"{current_time}_ekf_position_plot_excavator.png")
                )
    plt.close()
    # Plotting for worker
    plt.figure(figsize=(10, 6))
    plt.scatter(
        measurement_x_worker,
        measurement_z_worker,
        color="blue",
        label="Measurements",
        marker="o",
    )
    plt.scatter(
        estimated_x_kf_worker,
        estimated_z_kf_worker,
        color="red",
        label="KF Estimations",
        marker="^",
    )
    plt.plot(estimated_x_kf_worker, estimated_z_kf_worker, color="red")
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title("KF: Measurements vs Estimations for Positions for the Worker")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(os.path.join(worker_dir, f"{current_time}_kf_plot_worker.png"))
    plt.close()

    # Plotting for excavator positions
    plt.figure(figsize=(10, 6))
    plt.scatter(
        measurement_x_excavator,
        measurement_z_excavator,
        color="blue",
        label="Measurements",
        marker="o",
    )
    plt.scatter(
        estimated_x_ekf_excavator,
        estimated_z_ekf_excavator,
        color="red",
        label="EKF Estimations",
        marker="^",
    )
    plt.plot(estimated_x_ekf_excavator, estimated_z_ekf_excavator, color="red")
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title(
        "Extended KF: Measurements vs Estimations for Positions for the Excavator"  # noqa: E501
    )
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(excavator_dir, f"{current_time}_ekf_position_plot_excavator.png")  # noqa: E501
    )
    plt.close()

    # Plotting predicted positions for excavator
    plt.figure(figsize=(10, 6))
    plt.scatter(
        measurement_x_excavator,
        measurement_z_excavator,
        color="blue",
        label="Measured Positions",
        marker="o",
    )
    plt.scatter(
        predicted_x_excavator, predicted_z_excavator, color="green", label="Predicted Positions", marker="x"  # noqa: E501
    )
    plt.plot(predicted_x_excavator, predicted_z_excavator, color="green")
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title("Predicted vs Measured Positions for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir, f"{current_time}_predicted_position_plot_excavator.png"  # noqa: E501
        )
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(measured_vx)),
        measured_vx,
        color="blue",
        label="Measured VX",
        marker="o",
    )
    plt.scatter(
        range(len(predicted_vx)),
        predicted_vx,
        color="green",
        label="Predicted VX",
        marker="x",
    )
    plt.plot(range(len(predicted_vx)), predicted_vx, color="green")
    plt.xlabel("Time step")
    plt.ylabel("VX")
    plt.title("Predicted vs Measured X Velocities for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir, f"{current_time}_predicted_velocity_x_plot_excavator.png"  # noqa: E501
        )
    )
    plt.close()

    # Plotting Z velocity for excavator
    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(measured_vz)),
        measured_vz,
        color="blue",
        label="Measured VZ",
        marker="o",
    )
    plt.scatter(
        range(len(predicted_vz)),
        predicted_vz,
        color="green",
        label="Predicted VZ",
        marker="x",
    )
    plt.plot(range(len(predicted_vz)), predicted_vz, color="green")
    plt.xlabel("Time step")
    plt.ylabel("VZ")
    plt.title("Predicted vs Measured Z Velocities for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir, f"{current_time}_predicted_velocity_z_plot_excavator.png"  # noqa: E501
        )
    )
    plt.close()
    # Plotting estimated X velocity for excavator
    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(measured_vx)),
        measured_vx,
        color="blue",
        label="Measured VX",
        marker="o",
    )
    plt.scatter(
        range(len(estimated_vx)),
        estimated_vx,
        color="red",
        label="Estimated VX",
        marker="^",
    )
    plt.plot(range(len(estimated_vx)), estimated_vx, color="red")
    plt.xlabel("Time step")
    plt.ylabel("VX")
    plt.title("Estimated vs Measured X Velocities for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir, f"{current_time}_estimated_velocity_x_plot_excavator.png"  # noqa: E501
        )
    )
    plt.close()

    # Plotting estimated Z velocity for excavator
    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(measured_vz)),
        measured_vz,
        color="blue",
        label="Measured VZ",
        marker="o",
    )
    plt.scatter(
        range(len(estimated_vz)),
        estimated_vz,
        color="red",
        label="Estimated VZ",
        marker="^",
    )
    plt.plot(range(len(estimated_vz)), estimated_vz, color="red")
    plt.xlabel("Time step")
    plt.ylabel("VZ")
    plt.title("Estimated vs Measured Z Velocities for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir, f"{current_time}_estimated_velocity_z_plot_excavator.png"  # noqa: E501
        )
    )
    plt.close()
    # Plotting predicted vs estimated X velocity for excavator
    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(predicted_vx)),
        predicted_vx,
        color="green",
        label="Predicted VX",
        marker="x",
    )
    plt.scatter(
        range(len(estimated_vx)),
        estimated_vx,
        color="red",
        label="Estimated VX",
        marker="^",
    )
    plt.plot(range(len(predicted_vx)), predicted_vx, color="green")
    plt.plot(range(len(estimated_vx)), estimated_vx, color="red")
    plt.xlabel("Time step")
    plt.ylabel("VX")
    plt.title("Predicted vs Estimated X Velocities for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir,
            f"{current_time}_predicted_estimated_velocity_x_plot_excavator.png",  # noqa: E501
        )
    )
    plt.close()

    # Plotting predicted vs estimated Z velocity for excavator
    plt.figure(figsize=(10, 6))
    plt.scatter(
        range(len(predicted_vz)),
        predicted_vz,
        color="green",
        label="Predicted VZ",
        marker="x",
    )
    plt.scatter(
        range(len(estimated_vz)),
        estimated_vz,
        color="red",
        label="Estimated VZ",
        marker="^",
    )
    plt.plot(range(len(predicted_vz)), predicted_vz, color="green")
    plt.plot(range(len(estimated_vz)), estimated_vz, color="red")
    plt.xlabel("Time step")
    plt.ylabel("VZ")
    plt.title("Predicted vs Estimated Z Velocities for the Excavator")
    plt.legend(loc="lower right")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig(
        os.path.join(
            excavator_dir,
            f"{current_time}_predicted_estimated_velocity_z_plot_excavator.png",  # noqa: E501
        )
    )
    plt.close()


def evaluate_and_save_results():
    """Evaluate Kalman Filter results and save them."""
    global worker_evaluator, excavator_evaluator

    print("Evaluating Kalman Filter results...")

    # Update worker evaluator with all data points
    for i in range(len(measurement_x_worker)):
        worker_evaluator.add_data_point(
            true_state=[true_x_worker[i], true_z_worker[i]],
            measurement=[measurement_x_worker[i], measurement_z_worker[i]],
            estimated_state=[estimated_x_kf_worker[i], estimated_z_kf_worker[i]],
            predicted_state=[predicted_x_worker[i], predicted_z_worker[i]]
        )

    # Update excavator evaluator with all data points
    for i in range(len(measurement_x_excavator)):
        excavator_evaluator.add_data_point(
            true_state=[
                true_x_excavator[i], true_z_excavator[i]
            ],
            measurement=[
                measurement_x_excavator[i], measurement_z_excavator[i]
            ],
            estimated_state=[
                estimated_x_ekf_excavator[i], estimated_z_ekf_excavator[i]
            ],
            predicted_state=[
                predicted_x_excavator[i], predicted_z_excavator[i]
            ]
        )

    # Run full evaluation for worker
    worker_evaluator.run_evaluation("Worker Position")

    # Run full evaluation for excavator
    excavator_evaluator.run_evaluation("Excavator Position")


if __name__ == "__main__":
    try:
        # Run the main asyncio event loop
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Servers shut down gracefully.")
        # Calculate and print performance statistics
        elapsed_time = time.time() - start_time
        total_measurements = len(measurement_x_excavator)
        # Plot and save data
        plot_and_save_data()

        # Evaluate and save results
        evaluate_and_save_results()
