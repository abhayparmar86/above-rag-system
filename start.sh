#!/bin/bash
# Single entry point for launching the Above RAG System on any machine.
# Ensures a valid HTTPS cert exists for this machine's current IP(s) before
# starting the stack, so mic access never silently breaks on a new deployment.

CURRENT_IPS=$(hostname -I | tr ' ' '\n' | grep -v '^$')

cert_covers_current_ips() {
    [ -f certs/cert.pem ] || return 1
    local san
    san=$(openssl x509 -in certs/cert.pem -noout -ext subjectAltName 2>/dev/null)
    for ip in $CURRENT_IPS; do
        echo "$san" | grep -q "IP Address:$ip" || return 1
    done
    return 0
}

if cert_covers_current_ips; then
    echo "[start.sh] Existing certificate covers this machine's IP(s) — skipping regeneration."
else
    echo "[start.sh] Certificate missing or does not cover current IP(s) — generating..."
    bash generate_cert.sh
fi

echo "=================================================================="
echo " Above RAG System — open one of these links on the client device:"
for ip in $CURRENT_IPS; do
    echo "   https://${ip}:9001/static/index.html"
done
echo " (Self-signed cert — click through the one-time browser warning.)"
echo "=================================================================="

docker compose -f docker/docker-compose.yml up --build