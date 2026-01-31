#!/bin/bash
# Monitor System Health and Latency

THRESHOLD_MS=5
PROCESS_NAME="ExecutionEngine"

while true; do
    # Check if process is running
    p_count=$(pgrep -c $PROCESS_NAME)
    if [ $p_count -eq 0 ]; then
        echo "CRITICAL: $PROCESS_NAME is down!"
        # Send Alert (Simulated)
        # ./send_alert.sh "Engine Down"
        
        # Automatic restart attempt
        systemctl restart execution-engine
    fi

    # Check Latency Log for Spikes
    # Extract last minute high latency warnings
    error_count=$(tail -n 100 /var/log/trading/engine.log | grep "LATENCY_WARNING" | wc -l)
    
    if [ $error_count -gt 5 ]; then
        echo "WARNING: High Latency detected ($error_count spikes in last 100 lines)."
    fi

    sleep 10
done
