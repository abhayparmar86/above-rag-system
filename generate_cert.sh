#!/bin/bash
# Run this from rag_project/ (same level as the `docker/` folder).
# Generates a self-signed cert covering localhost + every local IPv4 address,
# so it stays valid even if the LAN IP changes before the demo.
set -e

mkdir -p certs

IPS=$(hostname -I | tr ' ' '\n' | grep -v '^$')
SAN_ENTRIES="DNS:localhost,IP:127.0.0.1"
for ip in $IPS; do
    SAN_ENTRIES="${SAN_ENTRIES},IP:${ip}"
done

echo "Generating cert for: ${SAN_ENTRIES}"

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem \
    -out certs/cert.pem \
    -subj "/CN=above-rag-demo" \
    -addext "subjectAltName=${SAN_ENTRIES}"

echo "Done. certs/key.pem and certs/cert.pem created."