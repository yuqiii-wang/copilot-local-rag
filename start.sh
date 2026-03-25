#!/bin/bash

# Kill processes on ports 8091 and 8092
for port in 8091 8092; do
    pids=$(lsof -t -i:$port 2>/dev/null)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
done

# Build extension
cd repo-ask
npm install
npm run compile
# npx vsce package

# Start servers (on ports 8091 and 8092)
cd dummy-servers
python -m pip install -r requirements.txt
nohup python confluence_server.py &
nohup python jira_server.py &


