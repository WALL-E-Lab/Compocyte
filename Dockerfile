# --- build stage ---
FROM python:3.14-slim-bookworm AS base
WORKDIR /app
RUN apt-get update; \
	apt-get install -y gcc g++
COPY . .
RUN pip install -e ".[dev]"

# --- test stage ---
FROM base AS test
RUN pytest

# --- production stage ---
#FROM base AS prod
#ENTRYPOINT []
#CMD ["your_tool"]