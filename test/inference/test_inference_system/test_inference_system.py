# test_inference_system.py
import asyncio
from inference_manager import InferenceManager
from channel import Channel
import time

async def main():
    start_time = time.time()
    # Initialize the channel
    channel = Channel()

    # Define server URLs and port ranges
    server_url = [
        {"host": "localhost", "ports": [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]},  # Example ports
    ]
    port_ranges = {
        (0, 50): [8000, 8001, 8002, 8003],  # Agents 0-100 can use ports 8000-8002
        (49, 100): [8004, 8005, 8006, 8007],  # Agents 0-100 can use ports 8000-8002
    }

    # Initialize the InferenceManager
    manager = InferenceManager(
        channel=channel,
        model_type="mock-model",
        model_path="/path/to/mock/model",
        stop_tokens=["\n"],
        server_url=server_url,
        port_ranges=port_ranges,
        timeout=300
    )

    # Start the InferenceManager
    manager_task = asyncio.create_task(manager.run())

    # Number of test requests
    num_requests = 100  # Adjust as needed for "massive" testing

    # Function to send requests
    async def send_requests():
        for i in range(num_requests):
            action_info = f"Test message {i}"
            agent_id = i % 100  # Example agent IDs
            message_id = await channel.write_to_receive_queue(action_info, agent_id)
            print(f"Sent request {i} with message_id {message_id}")
            await asyncio.sleep(0.01)  # Slight delay between sends

    # Function to collect responses
    async def collect_responses():
        collected = 0
        while collected < num_requests:
            # For testing, we'll check all message IDs
            keys = await channel.send_dict.keys()
            for message_id in keys:
                message = await channel.read_from_send_queue(message_id)
                print(f"Received response for {message_id}: {message[1]}")
                collected += 1
            await asyncio.sleep(0.1)
        print("All responses collected.")
        # Stop the manager after collecting all responses
        await manager.stop()

    # Start sending requests and collecting responses concurrently
    await asyncio.gather(
        send_requests(),
        collect_responses(),
        manager_task
    )

    # Print final metrics
    metrics = manager.get_metrics()
    end_time = time.time()
    print("Final Metrics:", metrics)
    print(start_time-end_time)

if __name__ == "__main__":
    asyncio.run(main())