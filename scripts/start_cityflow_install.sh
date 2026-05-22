#!/bin/bash
PROJECT="/mnt/d/Duy/Docs/School/CS106 - Trí tuệ nhân tạo/Đồ án/LLMLight"
chmod +x "$PROJECT/scripts/install_cityflow.sh"
nohup bash "$PROJECT/scripts/install_cityflow.sh" > /tmp/cityflow_nohup.out 2>&1 &
PID=$!
echo "Started CityFlow build PID: $PID"
sleep 2
ps -p "$PID" > /dev/null && echo "Confirmed running" || { echo "Failed to start"; cat /tmp/cityflow_nohup.out; }
