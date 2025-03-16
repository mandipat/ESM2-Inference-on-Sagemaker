#!/bin/bash
set -e

if [ "$1" = "serve" ]; then
  echo "Starting the model inference server..."
  python app.py  # Starts your Flask (or any other) server
else
  exec "$@"
fi
