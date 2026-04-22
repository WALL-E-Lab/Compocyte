# --- build stage ---
FROM python:3.14-slim-bookworm AS base
WORKDIR /app
COPY . .
RUN apt-get update; \
	apt-get install -y gcc g++
RUN pip install -e ".[dev]"

# --- test stage ---
FROM base AS test
RUN pytest tests/

# --- production stage ---
#FROM base AS prod
#ENTRYPOINT []
#CMD ["your_tool"]