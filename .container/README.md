# Install OASIS with Docker

Docker provides a consistent and isolated environment to build, run, and develop oasis without worrying about system dependencies. This guide walks you through setting up oasis using Docker, running it, and developing with it.

## Prerequisites

- Docker：https://docs.docker.com/engine/install/
- Docker Compose：https://docs.docker.com/compose/install/

## Configure Environment

Before starting the container, you need to navigate into the
[.container](../.container) folder and create a `.env` file **with your own
API keys**, so that these keys will be present in the environment variables of
the container, which will later be used by OASIS.

```bash
cd .container

# Create your own .env file by copying the example
cp .env.example .env

# Edit .env to add your API keys or custom settings
```

## Start Container

To build and start the development container:

```bash
docker compose up -d
```

This will:

- Build the image oasis:localdev
- Start the container oasis-localdev
- Set up all necessary dependencies

After the build is complete, you can verify the list of images and all running containers.

```bash
# List all Docker images
docker images
# Check running containers
docker ps
```

## Enter the Container

Once running, you can access the container like this:

```bash
docker compose exec oasis bash
```

You’ll now be inside the oasis dev environment.

From here, you can activate your virtual environment (if used) and run tests:

```bash
# Any other dev/test command
pytest
pre-commit run --all-files
```

## Save Your Progress

Your local code is volume-mounted into the container. That means any changes you make inside the container are reflected in your local project folder — no need to worry about losing your work.

## Exit, Stop and Delete the Container

You can simply press `Ctrl + D` or use the `exit` command to exit the
container.

After exiting the container, under normal cases the container will still be
running in the background. If you don't need the container anymore, you can
stop and delete the container with the following command.

```bash
docker compose down
```

## Online Images

For users who only want to have a quick tryout on OASIS, we also provide the
pre-built images on
[our GitHub Container Registry](<>).

## Pre-built Image (Optional)

If you only want to try oasis without setting up the build:

```bash

```
