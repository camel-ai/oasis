import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import logging
from typing import Dict, List, Tuple
import time

# Import the classes to test
from oasis.inference.inference_manager import InferencerManager, SharedMemory, PortManager

# Mock InferenceThread since we don't have the actual implementation
class MockInferenceThread:
    def __init__(self, model_path, server_url, stop_tokens, model_type, temperature, shared_memory):
        self.model_path = model_path
        self.server_url = server_url
        self.stop_tokens = stop_tokens
        self.model_type = model_type
        self.temperature = temperature
        self.shared_memory = shared_memory
        self.alive = True

    def run(self):
        pass

class MockChannel:
    def __init__(self):
        self.receive_queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()

    async def receive_from(self):
        return await self.receive_queue.get()

    async def send_to(self, message):
        await self.send_queue.put(message)

# Test configuration
TEST_SERVER_URL = [
    {
        "host": "localhost",
        "ports": [8000, 8001]
    }
]
TEST_MODEL_TYPE = "test_model"
TEST_MODEL_PATH = "/path/to/model"
TEST_STOP_TOKENS = ["<|end|>"]

class TestInferencerManager:
    @pytest.fixture
    def manager(self):
        # Setup mock channel
        channel = MockChannel()
        
        # Setup port ranges
        port_ranges = {(0, 10): [8000, 8001]}
        
        with patch('oasis.inference.inference_thread.InferenceThread', MockInferenceThread):
            # Create InferencerManager instance
            manager = InferencerManager(
                channel=channel,
                model_type=TEST_MODEL_TYPE,
                model_path=TEST_MODEL_PATH,
                stop_tokens=TEST_STOP_TOKENS,
                server_url=TEST_SERVER_URL,
                port_ranges=port_ranges,
                timeout=5
            )
            
            yield manager
            
            # Cleanup
            asyncio.run(manager.stop())

    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        assert len(manager.threads) == 2  # Should have two threads for ports 8000 and 8001
        assert manager.timeout == 5
        assert isinstance(manager.port_manager, PortManager)
        assert manager.metrics['total_requests'] == 0

    @pytest.mark.asyncio
    async def test_find_available_thread(self, manager):
        # Test finding available thread
        thread, port = await manager._find_available_thread(agent_id=1)
        assert thread is not None
        assert port in [8000, 8001]
        
        # Make all threads busy
        for thread in manager.threads.values():
            thread.shared_memory.Busy = True
            
        # Should return None when no threads available
        thread, port = await manager._find_available_thread(agent_id=1)
        assert thread is None
        assert port is None

    @pytest.mark.asyncio
    async def test_handle_new_request(self, manager):
        # Prepare test message
        test_message = ("msg_id_1", "test message", "1")
        await manager.channel.receive_queue.put(test_message)
        
        # Handle the request
        await manager._handle_new_request()
        
        # Verify thread assignment
        assigned_thread = None
        assigned_port = None
        for port, thread in manager.threads.items():
            if thread.shared_memory.Message_ID == "msg_id_1":
                assigned_thread = thread
                assigned_port = port
                break
                
        assert assigned_thread is not None
        assert assigned_port in [8000, 8001]
        assert manager.metrics['total_requests'] == 1

    @pytest.mark.asyncio
    async def test_process_completed_tasks(self, manager):
        # Simulate completed task
        test_port = 8000
        test_thread = manager.threads[test_port]
        test_thread.shared_memory.Message_ID = "msg_id_1"
        test_thread.shared_memory.Response = "test response"
        test_thread.shared_memory.Agent_ID = 1
        test_thread.shared_memory.Done = True
        
        # Process completed tasks
        await manager._process_completed_tasks()
        
        # Verify response was sent
        response = await manager.channel.send_queue.get()
        assert response == ("msg_id_1", "test response", 1)
        assert manager.metrics['successful_requests'] == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, manager):
        # Simulate timeout scenario
        test_port = 8000
        test_thread = manager.threads[test_port]
        test_thread.shared_memory.Busy = True
        test_thread.shared_memory.last_active = time.time() - manager.timeout - 1
        
        # Try to find available thread
        thread, port = await manager._find_available_thread(agent_id=1)
        
        # Should get the timed-out thread
        assert thread is test_thread
        assert port == test_port
        assert not thread.shared_memory.Busy

    @pytest.mark.asyncio
    async def test_metrics(self, manager):
        # Simulate successful request
        test_message = ("msg_id_1", "test message", "1")
        await manager.channel.receive_queue.put(test_message)
        await manager._handle_new_request()
        
        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['total_requests'] > 0
        assert isinstance(metrics['average_processing_time'], float)

if __name__ == '__main__':
    pytest.main(['-v'])