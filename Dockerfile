# Run a prebuilt release binary — no Rust toolchain, fast builds.
# The static musl binary runs on this glibc base without issues.
#
#   docker build -t anthropic-proxy .                              # latest full release
#   docker build -t anthropic-proxy --build-arg CHANNEL=prerelease . # latest pre-release (main)
#   docker build -t anthropic-proxy --build-arg VERSION=v2026.06.05+build.3 .  # pin a tag
#   docker buildx build --platform linux/amd64,linux/arm64 .       # multi-arch
#
# For building from source instead, use Dockerfile.source.
FROM debian:bookworm-slim

# Which build to fetch when VERSION is empty:
#   release    → the latest full release (default; pushed from the `release` branch)
#   prerelease → the latest pre-release (pushed from `main`)
ARG CHANNEL=release
# Pin an exact tag (overrides CHANNEL), e.g. v2026.06.05+build.3
ARG VERSION=
# Set automatically by `docker buildx` (amd64 / arm64).
ARG TARGETARCH

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl jq; \
    case "$TARGETARCH" in \
      amd64) TARGET=x86_64-unknown-linux-musl ;; \
      arm64) TARGET=aarch64-unknown-linux-musl ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    REPO="ExpTechTW/anthropic-proxy-rs"; \
    ASSET="anthropic-proxy-$TARGET.tar.gz"; \
    if [ -n "$VERSION" ]; then \
      URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"; \
    elif [ "$CHANNEL" = "prerelease" ]; then \
      TAG="$(curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=30" \
            | jq -r '[.[] | select(.draft==false and .prerelease==true)][0].tag_name')"; \
      [ -n "$TAG" ] && [ "$TAG" != "null" ] || { echo "no pre-release found for $REPO" >&2; exit 1; }; \
      URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"; \
    else \
      URL="https://github.com/$REPO/releases/latest/download/$ASSET"; \
    fi; \
    echo "Downloading $URL"; \
    curl -fsSL "$URL" | tar -xz -C /usr/local/bin; \
    chmod +x /usr/local/bin/anthropic-proxy; \
    apt-get purge -y --auto-remove curl jq; \
    rm -rf /var/lib/apt/lists/*

EXPOSE 3000

ENTRYPOINT ["anthropic-proxy"]
