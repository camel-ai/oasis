# Inference Manager Test Project

## Overview

The **Inference Manager Test Project** simulates and evaluates the inference management system in OASIS, which is designed to handle a large number of Large Language Model (LLM) requests efficiently. This project integrates key components to test the system’s ability to manage and process massive concurrent requests.

## Project Structure

```
inference-manager-test/
│
├── channel.py
├── mock_model_backend.py
├── inference_manager.py
├── test_inference_system.py
└── README.md
```

- **channel.py**: Manages communication between requests and responses.
- **mock_model_backend.py**: Simulates the behavior of an actual model backend.
- **inference_manager.py**: Handles the orchestration of inference threads and request processing.
- **test_inference_system.py**: Sends test requests and collects responses.
- **README.md**: Project documentation.

## Components

1. Channel

Handles the flow of messages between incoming requests and outgoing responses using asynchronous queues and dictionaries.

2. Mock Model Backend

Simulates processing of inference requests by introducing delays and returning mock responses without needing a real LLM model.

3. InferenceThread

Represents a worker that processes individual inference requests using the mock backend.

4. InferenceManager

Coordinates multiple InferenceThread instances, assigns incoming requests to available threads, and tracks performance metrics.

5. Test Harness

A script that sends a large number of test requests to the system and collects the responses to evaluate performance.

## Setup Instructions

### Prerequisites

The same with OASIS

### Run

```
# Modify the num_requests variable in ßtest_inference_system.py.
python test_inference_system.py
```

### what happens

- **Initialization**: Sets up the communication channel and inference manager with mock threads.
- **Sending Requests**: Sends a predefined number of test messages to simulate LLM requests.
- **Processing**: InferenceManager assigns requests to available threads for processing.
- **Collecting Responses**: Gathers and displays responses from the processed requests.
- **Metrics**: Displays performance metrics like total requests, successful and failed requests, and average processing time.
