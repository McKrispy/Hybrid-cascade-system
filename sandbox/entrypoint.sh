#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "HybridCascade: Self-Contained 7x24 Streaming Financial Decision Sandbox"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

# 1. Check environment variables
MISSING_KEYS=0
check_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "[!] WARNING: Missing environment variable: ${var_name}"
        MISSING_KEYS=$((MISSING_KEYS + 1))
    else
        echo "[+] Found ${var_name} (masked: ${!var_name:0:4}***)"
    fi
}

echo "[*] Checking API Credentials..."
check_env "TYPESAFE_API_KEY"
check_env "OPENAI_API_KEY"
check_env "DEEPSEEK_API_KEY"
check_env "APCA_API_KEY_ID"
check_env "APCA_API_SECRET_KEY"

if [ -z "${TELEMETRY_BEARER_TOKEN}" ]; then
    echo "[!] TELEMETRY_BEARER_TOKEN not set. Using default development token."
    export TELEMETRY_BEARER_TOKEN="change_me_telemetry_bearer_token"
fi

if [ $MISSING_KEYS -gt 0 ]; then
    echo "[!] Notice: $MISSING_KEYS API key(s) are missing."
    echo "[!] Sandbox will fall back to REPLAY_MODE using local historical events."
fi

# 2. Check network connectivity & geo-blocks (Binance 451, OpenAI block)
echo "[*] Probing API connectivity..."

# Probing Binance for HTTP 451 geo-restrictions
BINANCE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.binance.com/api/v3/ping || true)
if [ "$BINANCE_STATUS" = "451" ]; then
    echo "[!] WARNING: Binance returned HTTP 451 (Geo-blocked in this region/IP)."
    echo "[!] Alpaca Markets will serve as the primary market data feed."
elif [ "$BINANCE_STATUS" = "200" ]; then
    echo "[+] Binance Public API connectivity OK (HTTP 200)."
else
    echo "[~] Binance API probe status: $BINANCE_STATUS"
fi

# Probing TypeSafe
TYPESAFE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.typesafe.ai/health || true)
echo "[+] TypeSafe API probe response: ${TYPESAFE_STATUS:-reachable}"

# Probing OpenAI
OPENAI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models || true)
echo "[+] OpenAI API probe response: ${OPENAI_STATUS:-reachable}"

echo "[*] Initialization complete. Starting application..."
exec "$@"

