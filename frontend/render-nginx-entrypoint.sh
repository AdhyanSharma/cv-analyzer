#!/bin/sh
set -eu

# Render supplies BACKEND_HOSTPORT from the API service's private network.
# For local Docker Compose, preserve the existing service name.
export BACKEND_HOSTPORT="${BACKEND_HOSTPORT:-api:8000}"

exec /docker-entrypoint.sh "$@"
