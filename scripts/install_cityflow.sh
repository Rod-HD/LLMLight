#!/bin/bash
# Build and install CityFlow into the venv.
# Logs to /tmp/cityflow_install.log
set -e
PROJECT="/mnt/d/Duy/Docs/School/CS106 - Trí tuệ nhân tạo/Đồ án/LLMLight"
LOG=/tmp/cityflow_install.log
echo "=== CityFlow build started: $(date) ===" > "$LOG"
cd "$PROJECT/CityFlow"
"$PROJECT/venv/bin/pip" install --no-cache-dir . >> "$LOG" 2>&1
RC=$?
echo "" >> "$LOG"
echo "=== CityFlow build done: $(date) rc=$RC ===" >> "$LOG"
if [ $RC -eq 0 ]; then
    echo "" >> "$LOG"
    echo "=== Smoke test: import cityflow ===" >> "$LOG"
    "$PROJECT/venv/bin/python" -c "import cityflow; print('cityflow version:', getattr(cityflow, '__version__', 'unknown')); print('cityflow location:', cityflow.__file__)" >> "$LOG" 2>&1
fi
exit $RC
