#!/bin/bash
# Wrapper to start install_heavy_deps.sh in background with nohup.
# Outputs the PID so we can monitor.
PROJECT="/mnt/d/Duy/Docs/School/CS106 - Trí tuệ nhân tạo/Đồ án/LLMLight"
chmod +x "$PROJECT/scripts/install_heavy_deps.sh"
rm -f /tmp/heavy_deps_install.log
nohup bash "$PROJECT/scripts/install_heavy_deps.sh" > /tmp/heavy_install_nohup.out 2>&1 &
PID=$!
echo "Started PID: $PID"
sleep 2
if ps -p "$PID" > /dev/null; then
    echo "Confirmed running"
else
    echo "Failed to start, check /tmp/heavy_install_nohup.out:"
    cat /tmp/heavy_install_nohup.out 2>&1 | head -10
fi
