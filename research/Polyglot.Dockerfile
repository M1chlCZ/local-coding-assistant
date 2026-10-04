FROM docker.io/library/golang@sha256:966278043a40889499db9b0cd196fc789c37c385d41bd9a10cb1e7764af60cdc AS golang
FROM docker.io/library/rust@sha256:37f847a196b12e0c6298d265b99075a8bc9176f2bb22e434b4340ab7342ca3af AS rust
FROM docker.io/library/node@sha256:5a750d3be5e5c80275f8c9a5367c3aed99c2875656590c8d0701c7ee687f5f0a AS node
FROM docker.io/library/dart@sha256:4521dc04b39acb70b98be7b018ab0f3161a07346a6eb138bbf895211ca9cf6f3 AS dart
FROM python@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY --from=golang /usr/local/go /usr/local/go
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=dart /usr/lib/dart /usr/lib/dart
ENV PATH="/usr/local/go/bin:/usr/local/cargo/bin:/usr/lib/dart/bin:/usr/local/lib/node_modules/typescript/bin:${PATH}"
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
RUN node /usr/local/lib/node_modules/npm/bin/npm-cli.js install -g typescript@5.9.3 @types/node@24.10.1
RUN mkdir -p /opt/go-cache && printf 'package main\nfunc main(){}\n' >/tmp/warm.go && GOCACHE=/opt/go-cache go build -o /tmp/warm /tmp/warm.go && chmod -R a+rX /opt/go-cache
WORKDIR /workspace
