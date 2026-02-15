#!/bin/sh

APP="/run/main.py"

if [ ! -f "${APP}" ]; then
    echo "ERROR: main.py does not exist at ${APP}" >&2
    exit 1
fi

exec /opt/venv/bin/python "${APP}"
