#!/bin/sh
# Entry point for the full image:
#   train   run the whole pipeline (src/train.py)
#   serve   start the API on $PORT (needs a trained model in models/)
#   test    run the tests
#   anything else runs as a command
set -e
case "$1" in
  train) exec python -m src.train ;;
  serve) exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}" ;;
  test)  exec pytest -v ;;
  *)     exec "$@" ;;
esac