# Run a prebuilt release binary — no Rust toolchain, fast builds.
# The static musl binary runs on this glibc base without issues.
#
#   docker build -t anthropic-proxy .                          # latest full release
#   docker build -t anthropic-proxy --build-arg VERSION=v2026.06.05+build.3 .
#   docker buildx build --platform linux/amd64,linux/arm64 .   # multi-arch
#
# For building from source instead, use Dockerfile.source.
FROM debian:bookworm-slim

# Leave empty for the latest full release, or pin a tag like v2026.06.05+build.3
# (use a pre-release tag here — `latest` only resolves to full releases).
ARG VERSION=
# Set automatically by `docker buildx` (amd64 / arm64).
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in \
         amd64) TARGET=x86_64-unknown-linux-musl ;; \
         arm64) TARGET=aarch64-unknown-linux-musl ;; \
         *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
       esac \
    && BASE="https://github.com/ExpTechTW/anthropic-proxy-rs/releases" \
    && if [ -n "$VERSION" ]; then \
         URL="$BASE/download/$VERSION/anthropic-proxy-$TARGET.tar.gz"; \
       else \
         URL="$BASE/latest/download/anthropic-proxy-$TARGET.tar.gz"; \
       fi \
    && echo "Downloading $URL" \
    && curl -fsSL "$URL" | tar -xz -C /usr/local/bin \
    && chmod +x /usr/local/bin/anthropic-proxy

EXPOSE 3000

ENTRYPOINT ["anthropic-proxy"]
