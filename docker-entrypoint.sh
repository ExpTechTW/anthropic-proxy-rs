#!/bin/sh
# Launch the co-located open-websearch backend (optionally behind a pool of SSH egress
# proxies), then hand off to the proxy.
#
# open-websearch is the local web-search server the proxy uses to emulate Anthropic's
# server-side web_search/web_fetch tools for models that can't browse. It speaks MCP over
# Streamable HTTP on :3100. Engine + mode are chosen per-call by the proxy (DuckDuckGo +
# request), so here it only needs MODE=http and its own port — set explicitly so it does NOT
# inherit the proxy's PORT (e.g. 3000) and collide with it.
#
# Optional egress rotation (avoids single-IP rate-limiting): if a proxy-list file is mounted,
# we open one reconnecting SSH SOCKS tunnel per `user@host[:port] password` line, then front them with
# glider (round-robin + health checks) as ONE HTTP proxy, and point open-websearch at it via
# USE_PROXY/PROXY_URL. With no file, open-websearch searches directly (unchanged behaviour).

PROXY_FILE="${WEBSEARCH_SSH_PROXY_FILE:-/etc/websearch-ssh-proxies.txt}"
PROXY_LISTEN_PORT="${WEBSEARCH_PROXY_PORT:-7890}"

# Keep one SOCKS tunnel alive, reconnecting on drop. sshpass wraps ssh directly (autossh +
# sshpass is unreliable — the background fork loses the password handoff); the loop re-supplies
# the password each reconnect.
start_tunnel() {
    _target="$1"
    _pass="$2"
    _sport="$3"
    _lport="$4"
    while true; do
        sshpass -p "$_pass" ssh -N -D "127.0.0.1:$_lport" -p "$_sport" \
            -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes -o ConnectTimeout=10 "$_target"
        echo "egress tunnel $_target down; reconnecting in 5s" >&2
        sleep 5
    done
}

setup_ssh_egress() {
    [ -f "$PROXY_FILE" ] || return 1
    i=0
    forwards=""
    # Read "user@host[:port] password" lines (blank lines and #comments ignored).
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in '' | \#*) continue ;; esac
        target=$(printf '%s\n' "$line" | awk '{print $1}')
        pass=$(printf '%s\n' "$line" | awk '{print $2}')
        [ -n "$target" ] && [ -n "$pass" ] || continue
        sport=22
        case "$target" in *:*)
            sport=${target##*:}
            target=${target%:*}
            ;;
        esac
        lport=$((10800 + i))
        start_tunnel "$target" "$pass" "$sport" "$lport" &
        forwards="$forwards -forward socks5://127.0.0.1:$lport"
        i=$((i + 1))
        echo "  egress[$i]: $target -> socks5 127.0.0.1:$lport"
    done <"$PROXY_FILE"

    [ "$i" -gt 0 ] || return 1

    # Give the tunnels a moment to come up before glider's first health check.
    sleep 3
    # glider: one HTTP proxy that load-balances (round-robin) across the SSH SOCKS tunnels and
    # drops unhealthy ones via the periodic check.
    # shellcheck disable=SC2086
    glider -listen "http://127.0.0.1:$PROXY_LISTEN_PORT" $forwards \
        -strategy rr -check http://www.gstatic.com/generate_204 -checkinterval 30 &
    echo "websearch egress: $i SSH proxies via glider http://127.0.0.1:$PROXY_LISTEN_PORT"
    return 0
}

if setup_ssh_egress; then
    export USE_PROXY=true
    export PROXY_URL="http://127.0.0.1:$PROXY_LISTEN_PORT"
fi

MODE=http PORT="${WEBSEARCH_PORT:-3100}" open-websearch &

# Hand PID 1 to the proxy so signals/exit propagate normally; open-websearch (and any egress
# helpers) die with the container. If they crash, web search degrades but the proxy keeps serving.
exec anthropic-proxy "$@"
