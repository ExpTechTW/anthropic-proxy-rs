#!/usr/bin/env python3
# Periodically refresh every library indexed in docs-mcp (re-scrapes only changed pages, so it's
# cheap when nothing changed). docs-mcp has no built-in scheduler — run this from cron.
#
# Cron example — daily at 04:00, running on the same Docker network as docs-mcp:
#
#   0 4 * * * /usr/bin/docker run --rm --network <project>_trem-net \
#       -v /home/USER/refresh-docs.py:/r.py:ro python:3-slim python /r.py \
#       >> /home/USER/docs-refresh.log 2>&1
#
# (Replace <project>_trem-net with your compose network — `docker network ls`. For weekly use
#  `0 4 * * 0`; every 6 hours `0 */6 * * *`.) Override the endpoint with DOCS_MCP_URL.

import json, os, time, urllib.request

URL = os.environ.get("DOCS_MCP_URL", "http://docs-mcp:6280/mcp")


def call(method, params=None, rid=1):
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json",
                                          "accept": "application/json, text/event-stream"})
    text = urllib.request.urlopen(req, timeout=120).read().decode()
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return json.loads(text)


def tool(name, args):
    r = call("tools/call", {"name": name, "arguments": args}, rid=2)
    try:
        return r["result"]["content"][0]["text"]
    except Exception:
        return json.dumps(r)[:150]


print(time.strftime("%Y-%m-%d %H:%M:%S"), "docs refresh start")
call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "refresh", "version": "1"}})
libs = [l.strip()[2:].strip()
        for l in tool("list_libraries", {}).splitlines()
        if l.strip().startswith("- ")]
print("libraries:", libs)
for lib in libs:
    print(" ", lib, "->", tool("refresh_version", {"library": lib})[:120])
print(time.strftime("%Y-%m-%d %H:%M:%S"), "docs refresh queued")
