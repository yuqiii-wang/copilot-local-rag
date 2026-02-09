#!/bin/bash
npm install
npm run compile
npx vsce package

cd backend
pip install -r requirements.txt
python app.py

