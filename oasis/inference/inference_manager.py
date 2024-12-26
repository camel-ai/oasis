# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import sys

# Assuming InferenceThread and SharedMemory are defined elsewhere
from oasis.inference.inference_thread import InferenceThread

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SharedMemory:
    """Using dataclass for optimized memory usage and access efficiency"""
    Message_ID: Optional[str] = None
    Message: Optional[str] = None
    Agent_ID: Optional[int] = None
    Response: Optional[str] = None
    Done: bool = False
    Busy: bool = False
    Working: bool = False
    last_active: float = field(default_factory=time.time)  # Record last active time

class PortManager:
    """Class to manage port allocations"""
    def __init__(self, port_ranges: Dict[Tuple[int, int], range]):
        self.port_ranges = port_ranges
        self.agent_to_ports: Dict[int, List[int]] = defaultdict(list)
        self._initialize_mappings()

    def _initialize_mappings(self):
        """Initialize agent_id to port mappings"""
        for (start_id, end_id), ports in self.port_ranges.items():
            for agent_id in range(start_id, end_id + 1):
                self.agent_to_ports[agent_id].extend(ports)

    def get_ports_for_agent(self, agent_id: int) -> List[int]:
        """Get available ports for a given agent_id"""
        return self.agent_to_ports.get(agent_id, [])

class InferencerManager:
    def __init__(
        self,
        channel,
        model_type: str,
        model_path: str,
        stop_tokens: List[str],
        server_url: List[Dict],
        port_ranges: Optional[Dict[Tuple[int, int], List[int]]] = None,
        timeout: int = 300  # Timeout in seconds
    ):
        self.channel = channel
        self.threads: Dict[int, InferenceThread] = {}
        self.lock = asyncio.Lock()  # Use asyncio.Lock for async synchronization
        self.stop_event = asyncio.Event()
        self.count = 0
        self.timeout = timeout

        # Default configuration: all agents can access all ports
        if port_ranges is None:
            # Extract all ports from server_url
            all_ports = []
            for url_config in server_url:
                all_ports.extend(url_config.get("ports", []))
            
            # Create default configuration where all agents (0 to max int) can access all ports
            port_ranges = {(0, sys.maxsize): all_ports}

        # Initialize PortManager
        self.port_manager = PortManager(port_ranges)

        # Initialize threads
        self._initialize_threads(server_url, model_type, model_path, stop_tokens)

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0
        }

        # ThreadPoolExecutor for running blocking operations
        self.executor = ThreadPoolExecutor(max_workers=len(self.threads))

    def _initialize_threads(self, server_url, model_type, model_path, stop_tokens):
        """Initialize inference threads"""
        for url_config in server_url:
            host = url_config["host"]
            for port in url_config["ports"]:
                try:
                    _url = f"http://{host}:{port}/v1"
                    shared_memory = SharedMemory()
                    thread = InferenceThread(
                        model_path=model_path,
                        server_url=_url,
                        stop_tokens=stop_tokens,
                        model_type=model_type,
                        temperature=0.0,
                        shared_memory=shared_memory,
                    )
                    self.threads[port] = thread
                except Exception as e:
                    logger.error(f"Failed to initialize thread for port {port}: {e}")

    async def _find_available_thread(self, agent_id: int) -> Tuple[Optional[InferenceThread], Optional[int]]:
        """Find an available thread for the given agent_id"""
        available_ports = self.port_manager.get_ports_for_agent(agent_id)
        current_time = time.time()

        for port in available_ports:
            thread = self.threads.get(port)
            if thread is None:
                continue

            async with self.lock:
                if (not thread.shared_memory.Busy or 
                    (current_time - thread.shared_memory.last_active > self.timeout)):
                    if thread.shared_memory.Busy:
                        thread.shared_memory = SharedMemory()
                        logger.warning(f"Reset thread on port {port} due to timeout")
                    
                    return thread, port
        
        return None, None

    async def _process_completed_tasks(self):
        """Process completed inference tasks"""
        for port, thread in self.threads.items():
            async with self.lock:
                if thread.shared_memory.Done:
                    try:
                        await self.channel.send_to(
                            (thread.shared_memory.Message_ID,
                             thread.shared_memory.Response,
                             thread.shared_memory.Agent_ID))
                        
                        # Update metrics
                        self.metrics['successful_requests'] += 1
                        logger.debug(f"Processed completed task on port {port}")

                    except Exception as e:
                        logger.error(f"Error sending response for port {port}: {e}")
                        self.metrics['failed_requests'] += 1
                    finally:
                        # Reset thread state
                        thread.shared_memory = SharedMemory()
                        thread.shared_memory.last_active = time.time()

    async def _handle_new_request(self):
        """Handle new incoming requests"""
        try:
            message = await asyncio.wait_for(self.channel.receive_from(), timeout=0.1)
        except asyncio.TimeoutError:
            # No new message received within the timeout
            return
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return

        agent_id = int(message[2])
        start_time = time.time()

        available_thread, port = await self._find_available_thread(agent_id)
        
        if available_thread:
            async with self.lock:
                try:
                    available_thread.shared_memory.Message_ID = message[0]
                    available_thread.shared_memory.Message = message[1]
                    available_thread.shared_memory.Agent_ID = message[2]
                    available_thread.shared_memory.Busy = True
                    available_thread.shared_memory.last_active = time.time()
                    
                    self.count += 1
                    self.metrics['total_requests'] += 1
                    
                    # Update average processing time
                    processing_time = time.time() - start_time
                    self.metrics['average_processing_time'] = (
                        (self.metrics['average_processing_time'] * (self.count - 1) + processing_time)
                        / self.count
                    )
                    
                    logger.info(f"Assigned message {self.count} to port {port} for agent {agent_id}")

                    # Start the inference in a separate thread
                    asyncio.create_task(
                        self._run_inference(available_thread, message)
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing request for agent {agent_id}: {e}")
                    self.metrics['failed_requests'] += 1
                    # Requeue the message on failure
                    await self.channel.receive_queue.put(message)
        else:
            # No available threads; requeue the message
            await self.channel.receive_queue.put(message)

    async def _run_inference(self, thread: InferenceThread, message: Tuple[str, str, str]):
        """Run inference in a separate thread and update shared_memory"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                thread.run,  # Assuming `run` is a blocking method
            )
            async with self.lock:
                thread.shared_memory.Done = True
        except Exception as e:
            logger.error(f"Inference error on thread {thread.server_url}: {e}")
            async with self.lock:
                thread.shared_memory.Done = True
                thread.shared_memory.Response = f"Error: {e}"
        finally:
            thread.shared_memory.Busy = False
            thread.shared_memory.last_active = time.time()

    async def run(self):
        """Main run loop"""
        # Start all inference threads
        for port, thread in self.threads.items():
            # Start each thread in the ThreadPoolExecutor
            asyncio.get_event_loop().run_in_executor(self.executor, thread.run)

        # Create background tasks
        process_tasks_task = asyncio.create_task(self._process_completed_tasks_loop())
        handle_requests_task = asyncio.create_task(self._handle_requests_loop())

        try:
            await asyncio.wait(
                [process_tasks_task, handle_requests_task],
                return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            logger.info("Inference manager run task cancelled")
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
        finally:
            await self.stop()

    async def _process_completed_tasks_loop(self):
        """Continuously process completed tasks"""
        while not self.stop_event.is_set():
            await self._process_completed_tasks()
            await asyncio.sleep(0.1)  # Adjust as needed

    async def _handle_requests_loop(self):
        """Continuously handle incoming requests"""
        while not self.stop_event.is_set():
            await self._handle_new_request()
            await asyncio.sleep(0.1)  # Adjust as needed

    async def stop(self):
        """Stop all inference threads and perform cleanup"""
        self.stop_event.set()
        for thread in self.threads.values():
            thread.alive = False  # Ensure threads exit their run loops
        
        self.executor.shutdown(wait=True)
        
        # Log final metrics
        logger.info(f"Final metrics: {self.metrics}")

    def get_metrics(self) -> dict:
        """Retrieve performance metrics"""
        return self.metrics