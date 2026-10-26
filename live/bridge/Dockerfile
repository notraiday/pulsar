FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY bridge.py instructions.example.txt instructions.gpt-live.example.txt routes.example.json /app/
COPY livebridge /app/livebridge
# Keys come from the environment (--env-file); routes.json and instructions.txt are re-read for every dialog, so mount a
# directory (not single files) and point LIVE_ROUTES / LIVE_INSTRUCTIONS at it to edit them without a restart.
CMD ["python", "-u", "/app/bridge.py"]
