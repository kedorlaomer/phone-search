#!/data/data/com.termux/files/usr/bin/bash
# Start llama-server with Qwen3.5-4B (Termux). Idempotent: does nothing if it's already up.
#
# Qwen3.5-4B Q4_0, measured on the phone (prompt / generation tok/s):
# BACKEND=npu (default)  127 / 14. Needs the build from llama.cpp PR #29740 (integer HMX) in ~/npu-int: this v73
#                        NPU's matrix unit has no FP16 mode, so stock llama.cpp returns garbage on it.
# BACKEND=cpu            67 / 11 (stock upstream build in ~/npu/llama.cpp)
# BACKEND=gpu            141 / 6, but free RAM dipped to ~1.7 GB (risk: Android kills Termux)
set -eu
BACKEND=${BACKEND:-npu}
if [ "$BACKEND" = npu ]; then LLAMA=${LLAMA:-$HOME/npu-int}; else LLAMA=${LLAMA:-$HOME/npu/llama.cpp}; fi
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-Q4_0.gguf}
PORT=${PORT:-8080}
CTX=${CTX:-16384}
LOG=${LOG:-$HOME/phone-search/llm.log}

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  # Reuse it only if it's the build we asked for; otherwise replace it.
  running=$(for p in $(pgrep -f "bin/llama-server "); do tr '\0' ' ' < /proc/$p/cmdline; done)
  if [[ "$running" == "$LLAMA/bin/llama-server "* ]]; then
    echo "llama-server ($BACKEND) already running on :$PORT"
    exit 0
  fi
  echo "replacing running llama-server: ${running:0:80}"
  for p in $(pgrep -f "bin/llama-server "); do kill "$p"; done
  for _ in $(seq 1 30); do pgrep -f "bin/llama-server " >/dev/null || break; sleep 1; done
fi

export LD_LIBRARY_PATH=$LLAMA/lib:/vendor/lib64 ADSP_LIBRARY_PATH=$LLAMA/lib
case "$BACKEND" in
  cpu) DEV=(-dev none -t 6) ;;
  gpu) DEV=(-dev GPUOpenCL -ngl 99 -t 4) ;;
  npu) DEV=(-dev HTP0 -ngl 99 -t 4) ;;
  *) echo "unknown BACKEND=$BACKEND (cpu|gpu|npu)" >&2; exit 2 ;;
esac

nohup "$LLAMA/bin/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" "${DEV[@]}" \
  -c "$CTX" -np 1 --jinja --reasoning-format deepseek \
  > "$LOG" 2>&1 < /dev/null &
echo "starting llama-server ($BACKEND, pid $!), log: $LOG"
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready on :$PORT"; exit 0; }
  sleep 1
done
echo "llama-server did not come up within 120 s; see $LOG" >&2
exit 1
