import asyncio
import zmq
import zmq.asyncio
import json
import struct
import time
import signal


# Initialize global variables
start_time = time.time()
n_results_sent = 0
shutdown_event = asyncio.Event()


async def handle_client(reader, writer):
    """Handle client connections and process detection results."""
    global n_results_sent
    # Create ZeroMQ context and socket
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    try:
        while not shutdown_event.is_set():
            try:
                # Receive detection result with timeout
                detection_result = await asyncio.wait_for(
                    socket.recv_json(), timeout=1.0
                )
                print("received detection_result")
                print(
                    f"Received detection result for frame {detection_result.get('frame_number', 'unknown')}"
                )

                # Prepare the result for sending
                json_data = json.dumps(detection_result)
                framed_message = f"<START>{json_data}<END>".encode("utf-8")

                message_length = len(framed_message)
                try:
                    # Send message length and content
                    writer.write(struct.pack(">I", message_length))
                    await writer.drain()
                    writer.write(framed_message)
                    await writer.drain()
                    n_results_sent += 1
                    print(
                        f"Sent detection result for frame {detection_result.get('frame_number', 'unknown')} to Godot"
                    )
                    n_results_sent += 1
                except (ConnectionResetError, BrokenPipeError):
                    print("Client disconnected unexpectedly")
                    break
            except asyncio.TimeoutError:
                continue
            except (ConnectionResetError, BrokenPipeError):
                print("Client disconnected")
                break
    except asyncio.CancelledError:
        print("Task was cancelled")
    except Exception as e:
        print(f"Error in handle_client: {e}")
    finally:
        # Clean up resources
        socket.close()
        try:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                print("Timed out waiting for writer to close")
            except Exception as e:
                print(f"Error while closing writer: {e}")
        except Exception as e:
            print(f"Error while closing connection: {e}")
        print("Result client connection closed")
        shutdown_event.set()  # Trigger shutdown when client disconnects


async def shutdown(signal, loop):
    """Handle graceful shutdown of the server."""
    print(f"Received exit signal {signal.name}...")
    shutdown_event.set()

    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main():
    """Main function to set up and run the server."""
    loop = asyncio.get_running_loop()
    # Set up signal handlers
    signals = (signal.SIGINT, signal.SIGTERM)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )

    # Start the server
    server = await asyncio.start_server(handle_client, "127.0.0.1", 9998)

    async with server:
        try:
            await asyncio.wait_for(server.serve_forever(), timeout=None)
        except asyncio.CancelledError:
            print("Server was cancelled")
        finally:
            server.close()
            await server.wait_closed()
            print("Result server shut down gracefully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Calculate and print average FPS
        elapsed_time = time.time() - start_time
        print(
            f"Average FPS result server: {n_results_sent / elapsed_time:.2f}"
        )
print(f"Total results sent: {n_results_sent}")
