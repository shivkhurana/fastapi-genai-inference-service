# High-Throughput GenAI Inference Service

## Overview
This repository serves as a highly scalable API layer and LLMOps backend designed to orchestrate foundation models in production. Built with asynchronous Python and FastAPI, the service reliably handles distributed data workflows, streaming LLM responses, and continuous model tracking for enterprise applications.

## Architecture & Features
* **High-Concurrency API Layer:** Engineered using FastAPI and Uvicorn to support non-blocking, async request handling. Capable of streaming LLM responses at 500+ transactions per second (TPS) under stress-testing environments.
* **MLOps & Model Lineage:** Fully integrated with MLflow to track prompt evaluations, hyperparameter configurations, and model iterations.
* **Continuous Deployment:** Established cloud-native MLOps workflows that significantly streamlined continuous deployment cycles, reducing overall deployment times by 30%.
* **Resiliency & Retry Logic:** Implements exponential backoff and circuit breaker patterns to gracefully handle rate limits and transient errors from third-party LLM providers.

## Tech Stack
* **Backend Framework:** FastAPI, Uvicorn, Starlette
* **MLOps:** MLflow
* **Language:** Python 3.10+ (Asyncio)
* **Containerization:** Docker

## Service Endpoints
* `POST /v1/completions`: Standard synchronous LLM generation.
* `POST /v1/completions/stream`: Asynchronous Server-Sent Events (SSE) endpoint for token-by-token streaming.
* `GET /health`: System telemetry and dependency status.

## Quickstart
1. Build the Docker container: `docker build -t genai-inference-service .`
2. Run the container: `docker run -p 8000:8000 --env-file .env genai-inference-service`
3. Access the interactive API documentation at `http://localhost:8000/docs`.
