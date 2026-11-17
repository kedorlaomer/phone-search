#!/data/data/com.termux/files/usr/bin/bash
# Start llama-server with Qwen3.5-4B (Termux). Idempotent: does nothing if it's already up.
#
# BACKEND=cpu (default)  correct output, 67 tok/s prompt / 11 tok/s generation, leaves most RAM free
# BACKEND=gpu            Adreno OpenCL: 141 / 6 tok/s, but free RAM dipped to ~1.7 GB (risk: Android kills Termux)
# BACKEND=npu            Hexagon with HMX disabled: 30 / 13 tok/s. With HMX enabled the v73 NPU returns garbage
#                        (53/82 q4_0 MUL_MAT tests fail), so GGML_HEXAGON_NHMX=0 is mandatory.
set -eu
LLAMA=${LLAMA:-$HOME/npu/llama.cpp}
MODEL=${MODEL:-$HOME/models/Qwen3.5-4B-Q4_0.gguf}
BACKEND=${BACKEND:-cpu}
PORT=${PORT:-8080}
CTX=${CTX:-16384}
LOG=${LOG:-$HOME/phone-search/llm.log}

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "llama-server already running on :$PORT"
  exit 0
fi

export LD_LIBRARY_PATH=$LLAMA/lib:/vendor/lib64 ADSP_LIBRARY_PATH=$LLAMA/lib
case "$BACKEND" in
  cpu) DEV=(-dev none -t 6) ;;
  gpu) DEV=(-dev GPUOpenCL -ngl 99 -t 4) ;;
  npu) export GGML_HEXAGON_NHMX=0; DEV=(-dev HTP0 -ngl 99 -t 4) ;;
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
