#!/bin/bash
cd repo-ask
npm install
npm run compile
npx vsce package

cd dummy-servers
pip install -r requirements.txt
nohup python confluence_server.py &
nohup python jira_server.py &

