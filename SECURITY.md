# Security

## Reporting a vulnerability

Please report anything that should not be public until it is fixed **privately**, through GitHub's private
vulnerability reporting: the repository's **Security** tab, **Report a vulnerability**
([direct link](https://github.com/Indras-Mirror/Xolotl/security/advisories/new)). Say what you found, how to reproduce it
(the request, the config keys involved, the Xolotl commit), and what an attacker gains. We answer there, fix it in
a release, and credit you in the advisory unless you ask us not to.

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/Indras-Mirror/Xolotl/issues) (upstream Strata's audit findings are collected in [Strata #544](https://github.com/Niko1221/Strata/issues/544)) or as a pull request, which we
review like our own code.

Supported: the latest commit on `main` (Xolotl has no releases). The server code below is Strata's, so the same
protections apply.

## Local models and your network

A model file (GGUF) is weights and metadata: Xolotl's C++ engines read it as data, and nothing in it runs as code.
Checked on 8 Nov 2026:

- **The engines have no network code.** `strata`, `xolotl-dense`, `ds41_generate` and `glm_generate` link no network
  library (no curl, no TLS) and open no internet socket: they talk to the server only through stdin / stdout. Watched
  while the 27B worked through agent tasks: zero inet sockets. `tools/xolotl/apparmor/xolotl-engines` makes that a
  rule the kernel enforces (one `sudo` to install). The one exception is `xolotl-lfm3`: next to another model
  (`--peer-vram`) it runs `curl` to that server's `/v1/vram` on localhost to share the GPU, so it is left out of the
  profile.
  **If you add network code to an engine** (any socket, curl, an HTTP fetch, even to localhost), the profile blocks it
  and that engine fails with a permission error (`EACCES`, "Permission denied" on `socket()`) or won't start. Check
  with `journalctl -k | grep 'profile="xolotl-engine"'` (DENIED lines name the engine and the operation). Then
  either keep the network part in the Python server, leave that engine out of the profile's name list (as with
  `xolotl-lfm3`), or remove the profile. Tested under the profile on 8 Nov 2026: DeepSeek-V4.1 fast, QFN45, the 27B
  and Flash-Next start and answer, `xol-lease` sleep / wake / unload / load works, and LFM3 (unconfined, with its
  speech and vision helpers) chats, speaks and parks / unparks over `--peer-vram`.
- **The server listens on `127.0.0.1`** unless you pass `--host`. It connects out only to what you configure (MCP
  servers in the run config, a voice backend) and to an image URL that a client request contains. Remote JSON-schema
  `$ref`s are never fetched. Chat templates from model files render in Jinja's immutable sandbox. Nothing loads
  pickles or `trust_remote_code`.
- **The real risk is tools.** A model can only act on the world through the tools an agent gives it. In QuetzaCodetl,
  shell commands, the JS REPL, web search and web fetch (outside a short list of documentation sites) all ask before
  they run, unless you allow them.

## What the server exposes

Strata runs a model on your PC and serves it over HTTP (`serve/server.py`). Out of the box it is reachable from
this PC only. The details and every setting are in [docs/DETAILS.md](docs/DETAILS.md) ("From other devices",
"Host names", "Web pages without an API key", "Tools from MCP servers").

- **Where it listens.** `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in `strata-<model>.json`) opens it to
  your network, and the server then warns when no API key is set.
- **API key.** `"api_key"` in the run config (or `STRATA_API_KEY`) is required on `/v1/*` and on every endpoint
  that shows the model's state, requests or answers (`/status`, `/metrics`, `/metrics/prometheus`, `/settings`,
  `/mcp`, `/props`, `/slots`, `/api/requests`, `/config`) and on every `POST`. It is compared in constant time.
  Set one before you open the
  server to your network or put a tunnel in front of it.
- **Host check (DNS rebinding, 0.1.38).** Without an API key the server answers only requests whose `Host` is a name
  it knows (`localhost`, an IP address, the address it listens on and, when it listens beyond this PC, this PC's
  name). Others get 403. `"allowed_hosts"` adds names; with an API key the key decides.
- **Origin check (0.1.38).** Without an API key, a browser `POST` to `/v1/*` from another web site's page (it carries
  an `Origin` header) gets 403 unless that origin is Strata's own page, `localhost`, an allowed host, or listed in
  `"trusted_origins"` / `"cors_origins"`. Changing settings (`/settings`, and `/config`: the few run config keys the
  page's Model settings may change - never the network, key, MCP or program keys), `/load` and `/unload` and the
  MCP tools are accepted only as JSON from Strata's own page (or `"trusted_origins"`), so a page elsewhere cannot
  change settings or run tools.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **MCP tools** are opt-in: only the servers you put in the run config, only for requests from Strata's own page
  that ask for them. They run with your user's rights, and the model decides when to call them.
- **The request monitor** (`/api-monitor`, which keeps the last prompts and answers in memory) is off unless
  `"api_monitor": true` is set.
- **The request log** (`--request-log PATH`, one line of timings and token counts per request) is off unless set. It
  writes no prompts, answers or keys, but it does keep each client's address and `User-Agent`.
  `"api_monitor": true` is set. Apart from it, the server keeps in its process memory the text and tokens of its
  last few prompts, whatever `"api_monitor"` says, to encode the next turn faster; no endpoint shows them, and they
  are gone when the server stops.

## The desktop app and the Terminal

Strata comes in three parts that share one engine: the **desktop app**, the **Terminal** (`strata`) and the **engine** that runs
the model. The section above is about the engine's HTTP server. This is what the other two add.

- **The app's host listens on this PC only.** The app serves its page from a local host process on `127.0.0.1` (port 8791 unless
  changed). A request to its `/api/` endpoints that carries a browser `Origin` from any other site gets 403. The app shell holds a
  random secret (32 characters or more, compared in constant time) that is needed to quit the host or tell it which window is open,
  and the host gives its own helper processes a second random secret. By default the models the app starts are served on this PC only.
- **Sharing is a separate, key-protected listener.** "Share with other devices" (and the Cloudflare tunnel) will not start until an
  API key is set. It opens a different listener that answers only `/v1/models`, `/v1/chat/completions` and `/v1/messages`, never the
  app's page or its host endpoints.
- **App and Terminal talk through an owner channel.** One owner process per installation starts and ends the engine, and a second
  client (the Terminal) attaches to it. On macOS and Linux that is a private Unix socket in a folder only you can open. On Windows it
  is a TCP listener on `127.0.0.1` that answers only requests carrying a 256-bit token kept in an owner-only folder. That is a weaker
  guarantee than a socket other users cannot open: on Windows the token is what protects it, as with any loopback service that has a
  token (`strata_control/owner_transport.py`). The bridge between the app shell and the host (used for the app update) is also loopback and protected by a token.
- **Starting a model and serving never download anything.** Downloads and updates are one explicit step: `strata update`, or "Update
  engine" and "Update Strata" in the app's Settings. Models and saved tuning are never replaced by an update.
- **Updates come from a signed feed.** The release publishes `strata-feed.json` and `strata-feed.json.sig` (Ed25519). An installed
  Strata checks the signature against the public key it carries (`strata_control/update_keys.json`), refuses a feed older than the last
  one it accepted (a replayed old feed) or expired, and checks every engine package against its SHA-256, its signed manifest and the
  card in your PC before it test-starts it and switches to it. Any failure keeps the engine you had. The app installer itself is signed
  with a separate key (Tauri's minisign key) that the app and the Terminal both check. The private keys are kept offline by the owner.
  See [packaging/UPDATES.md](packaging/UPDATES.md).

### How far to trust `curl | sh` and `irm | iex`

(This is upstream Strata's installer. Xolotl itself is built from source - see [AGENTS.md](AGENTS.md).)

The Terminal installs with one line:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Piping a script into a shell runs whatever the server sends, so it deserves the same care as running any downloaded program.
What the script does is short and written at its top: it fetches the Terminal archive named in the release's `install-manifest.txt`,
checks its SHA-256, unpacks it into your home folder (no administrator rights, nothing outside your home), checks the release's signed feed
with the Terminal it just unpacked and removes it again if that fails, and puts `strata` on your PATH.

Those checks have a limit. The checksum comes from the same release as the archive, and the public key that checks the feed arrives inside
that archive, so for the first install all trust rests on HTTPS and on the `Niko1221/Strata` GitHub release itself. The checks catch a
damaged or half-finished download and a feed that was not signed by the release key; they do not protect you if the release has been replaced
by someone who controls it. Later updates are checked against the key already on your PC, which is a stronger position.

If that is not enough: save the script (`curl -fsSLO https://github.com/Niko1221/Strata/releases/latest/download/install.sh`), read it, and
run it from the file; or install the desktop app or build from source instead and check the code you build.

Strata has not had an outside security audit yet.
What Strata downloads, when it uses the internet, what stays on your PC and how downloads are checked is written up in [docs/SAFETY.md](docs/SAFETY.md).
