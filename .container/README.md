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
- Create a Python 3.14 development environment
- Install the project and its development dependencies from `uv.lock`

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

You’ll now be inside the OASIS development environment. The image includes
uv, and the project environment lives at `/app/oasis/.venv`. Its `bin`
directory is on `PATH`, so `python`, `pip`, and project tools use that
environment by default.

### Run commands

```bash
# Run commands through uv (recommended)
uv run pytest
uv run pre-commit run --all-files

# The virtual environment is already on PATH, so these are also valid
python -m pytest
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
[our GitHub Container Registry](https://github.com/orgs/camel-ai/packages/container/package/oasis).

## Pre-built Image (Optional)

To try OASIS v0.2.6 without building the image locally:

```bash
docker pull ghcr.io/camel-ai/oasis:v0.2.6
```

From the `.container` directory, use the `.env` file configured above to
start an interactive shell:

```bash
docker run --rm -it --env-file .env ghcr.io/camel-ai/oasis:v0.2.6 bash
```

API keys are supplied at runtime through `.env`; they are not included in the
image. This command does not mount your local project, and changes made inside
the container are removed when you exit.
