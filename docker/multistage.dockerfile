# Stage 1: Build stage (uses the development image with build tools)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build

# Install Python and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends python3 build-essential 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy your application files and install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject-uv.toml,target=pyproject.toml \
    uv sync --no-install-project --no-dev --extra cu12

# Stage 2: Runtime stage (uses the lighter runtime image)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

# Install just Python (if not already present in the specific runtime image)
RUN apt-get update && apt-get install -y python3 --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application from the build stage
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
EXPOSE 5000
# Set the entry point for your application
CMD ["python3", "./main.py"]
