# Demo: exposing the live CastleRAG UI

`scripts/slurm/ui_live.slurm` boots the full backend (OmniEmbed `:8200`,
Qwen3-VL `:8201`, Qdrant) against the **existing** persisted index and serves the
Dash dashboard with the live `RagEngine`. It is non-destructive (never
preprocesses / embeds / indexes / creates collections, and aborts if the
collection looks emptied).

It has three exposure modes, selected by env vars at submit time:

| Mode | Trigger | Who can reach it | Auth |
|------|---------|------------------|------|
| **SSH tunnel** (default) | neither var set | anyone with a Snellius account | SSH |
| **Quick tunnel** | `PUBLIC_TUNNEL=1` | anyone with the link | HTTP basic auth (required) |
| **Named tunnel** | `NAMED_TUNNEL_TOKEN=…` | your email allowlist | Cloudflare Access |

All modes run `castlerag ui --require-live`, so the job fails loudly with the
exact missing dependency rather than silently serving the offline placeholder.

> **Why `--protocol http2`?** Snellius compute nodes firewall QUIC/UDP-7844
> (cloudflared's default), so the tunnel registers but the data path times out
> (HTTP 530). TCP/443 reaches the edge, so we force `--protocol http2`. Verified
> on `gpu_a100`: QUIC → 530, http2 → HTTP 200 end-to-end. This is baked into the
> script; nothing to set.

---

## SSH tunnel (default — for developers)

```bash
sbatch --account=gpuuva082 scripts/slurm/ui_live.slurm
# NODE is printed in logs/castle-ui-live_<jobid>.out, then from your laptop:
ssh -L 8050:<NODE>:8050 <user>@snellius.surf.nl
# open http://localhost:8050  (top-bar chip should read "live RAG")
```

Only works for people with a Snellius account. Not for external demo attendees.

---

## Named tunnel (recommended for a scheduled demo)

Stable, pre-shareable URL on your domain + Cloudflare Access (email allowlist).
No shared password. Requires a Cloudflare account with a domain (zone).

### One-time setup (Cloudflare Zero Trust dashboard)

1. **Networks → Tunnels → Create a tunnel** → type **Cloudflared** → name it
   (e.g. `castle-demo`) → **copy the connector token** (the long `eyJ…` string).
   No `cloudflared login` / cert needed on the node — the token is enough.
2. On that tunnel, **add a Public Hostname**:
   - Subdomain/domain: e.g. `castle-demo.<yourdomain>`
   - Service: **HTTP** → **`localhost:8050`**
3. **Access → Applications → Add → Self-hosted**, hostname
   `castle-demo.<yourdomain>`, then a policy: **Allow → Emails / Emails ending
   in** → your attendees' addresses.

### Run it (demo day)

```bash
sbatch --account=gpuuva082 \
  --export=ALL,NAMED_TUNNEL_TOKEN=eyJ...<your token>... \
  scripts/slurm/ui_live.slurm
```

The connector comes up and your **stable URL** `castle-demo.<yourdomain>` serves
the live UI. Attendees load it → Cloudflare Access prompts for their email →
only allowlisted ones get in. `scancel <jobid>` to stop; the hostname stays
reserved for next time.

- The **token is a secret** — pass it via `--export` at submit time (as above)
  or a file the job reads; never commit it.
- Basic auth is **optional** in named mode (Access already gates at the edge).
  Set `CASTLERAG_UI_BASIC_AUTH=user:pass` too only if you want defense-in-depth.

---

## Quick tunnel (fallback — no Cloudflare account needed)

Ephemeral `https://<random>.trycloudflare.com` URL, gated by HTTP basic auth.
The URL **rotates every run** and has no uptime guarantee, so share it at the
start of the demo and tear down after.

```bash
sbatch --account=gpuuva082 \
  --export=ALL,PUBLIC_TUNNEL=1,CASTLERAG_UI_BASIC_AUTH=demo:<password> \
  scripts/slurm/ui_live.slurm
# public URL prints in logs/castle-ui-live_<jobid>.out
```

The job **refuses to open the tunnel without `CASTLERAG_UI_BASIC_AUTH`** — the
app gates every request, so the live GPU backend is never exposed unauthenticated.

---

## Pre-flight checklist

- [ ] Confirm SURF/Snellius acceptable-use policy permits a public tunnel.
- [ ] `~/cloudflared` present (`curl -L .../cloudflared-linux-amd64 -o ~/cloudflared && chmod +x ~/cloudflared`), or set `CLOUDFLARED=<path>`.
- [ ] Existing index intact (`castle_multimodal_v1`, ≥25k points) — the job aborts otherwise.
- [ ] **Dry-run the full chain once before demo day** from an off-network machine.
- [ ] Bump `#SBATCH --time` to cover the demo window + buffer; submit early (GPU queue).
