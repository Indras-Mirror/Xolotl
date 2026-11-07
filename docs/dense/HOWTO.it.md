# Strata — howto operativo

Due percorsi, due binari, due famiglie di modelli. Non sono intercambiabili.

| | MoE | Dense |
|---|---|---|
| Binario | `engine/strata` | `build-dense/strata-dense` |
| Modelli | Qwen3.8-Flash-Next (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S), Coder IQ1_M | Qwen3.8-27B (GGUF unsloth, qualsiasi quant) |
| Architettura | `qwen4exp`, 48 layer × 512 expert, 2560 wide | `qwen35`, 64 layer, n_embd 5120, n_ff 17408 |
| Expert cache / profilo | sì | **no** (nessun expert) |
| Speculative decoding | sì (MTP + suffix drafter) | no |
| Pack necessario | sì (`tools/strata_pack.py`) | no, legge il GGUF così com'è |

`strata` rifiuta il GGUF denso al primo guard di `check_architecture`; `strata-dense` non sa cosa sia un expert.

---

## 1. Expert profile — cosa è e perché decide la velocità

### Il problema

Il modello ha **48 × 512 = 24576 expert**. Il router ne sceglie **10 per layer per token**. I blob non stanno tutti in VRAM.

Il profilo (`expert-profile.bin`) è un **piano di residenza statico**: una lista ordinata di coppie `(layer, expert)`. L'engine prende le prime N e le precarica in VRAM.

- **hit** → expert residente, calcolato sulla GPU (costo ~0)
- **miss** → calcolato sul CPU pool (~57 µs l'uno, vedi sotto) o streamato in PCIe

### Formato del file

Definito in [`include/strata/core/expert_cache.hpp:43`](../../include/strata/core/expert_cache.hpp:43), letto da [`src/core/expert_cache.cpp:12`](../../src/core/expert_cache.cpp:12):

```
offset  0  "STRP"                        magic, 4 byte
offset  4  version, n_layers, n_expert,
           slots, n_ranked               5 × uint32 little-endian
offset 24  n_ranked × (uint16 layer, uint16 expert)   la lista ordinata
           n_layers × n_expert × int32                tabella slot-per-expert (lookup)
```

L'engine usa **solo la lista ordinata**. La tabella di lookup è il lavoro interno del tool. Il campo `slots` è informativo: è il "built for N slots" che compare nel log.

Validazione ([`src/core/expert_cache.cpp:32`](../../src/core/expert_cache.cpp:32)): se `(n_layers, n_expert)` non coincide con la geometria del modello, l'engine si ferma con

```
read_expert_profile: <file> is 48x512 but this model is 64x256 -
it is a profile for a different artifact
```

### Il ranking conta più del numero di slot

Il routing MoE segue una legge di potenza. Il repo la modella così in [`src/program/generate.cpp:1998`](../../src/program/generate.cpp:1998):

```
massa instradata per rank r  ∝  (r+1)^-1.2
```

Esponente 1.2 < 1 → coda pesante: i pochi expert in cima si prendono quasi tutto il traffico. Quindi **copertura in numero di expert ≠ copertura in massa instradata**, ed è la massa che determina l'hit rate.

### La trappola del default, con i conti

[`tools/make_default_profile.py`](../../tools/make_default_profile.py) genera coppie in ordine **layer-major**: tutti i 512 expert della layer 0, poi layer 1, ecc. Utile solo come placeholder per far partire l'engine.

Con 9615 slot (quelli che stanno su una 3090/4090 con IQ3_S):

```
9615 ÷ 512 = 18.78
9615 = 18 × 512 + 399
```

| Layer | Expert resident | Copertura |
|---|---|---|
| 0 – 17 | 512 / 512 | 100% |
| 18 | 399 / 512 | 77.9% |
| 19 – 47 | **0 / 512** | **0%** |

Hit rate atteso (48 layer equivalenti):

```
h = (18 + 0.779) / 48 = 39.1%
```

Misurato: **43.7 – 44.4%**. Il +5% viene dal fatto che la layer 18 prende i *suoi* 399 expert più caldi.

Il danno strutturale: **29 layer su 48 con hit rate esattamente 0%**, qualunque sia il prompt.

Controfigura — profilo by-frequency, 9615 slot distribuiti su tutte le layer:

```
9615 ÷ 48 = 200.3 slot per layer
```

Copertura in massa per una layer, con Zipf esponente 1.2:

```
h = Σ_{r=1}^{200} r^-1.2  ÷  Σ_{r=1}^{512} r^-1.2
```

Somma parziale di ζ(1.2) ≈ 5.60; coda da 513 ≈ ∫ x^-1.2 dx = 5·513^-0.2 = 1.44; coda da 201 ≈ 5·200.5^-0.2 = 1.73:

```
denominatore = 5.60 − 1.44 = 4.16
numeratore   = 5.60 − 1.73 = 3.87
h = 3.87 / 4.16 = 93.0%
```

Misurato: **86 – 94%**.

| | layer-major | by-frequency |
|---|---|---|
| Slot | 9615 | 9615 |
| Expert resident | 39.1% | 39.1% |
| **Massa instradata coperta** | **39.1%** | **93.0%** |
| Hit rate misurato | 43.7–44.4% | 86–94% |

Stessi slot, stesso numero di expert in VRAM. Il 39% *giusto* di expert porta il 93% del traffico.

### Da hit rate a tok/s

Modello dai costi in [`include/strata/spec/controller.hpp:25`](../../include/strata/spec/controller.hpp:25), window T=4 (n = k+1 = 4):

```
step_ms = dense(n) + experts(n) + sync + draft

dense   = 11.0 × dense_ratio[3] = 11.0 × 1.45              = 15.95 ms
experts = 15.8 × distinct_ratio[3] × (1 − h) = 15.8 × 2.88 × (1 − h)
sync    = 2.4 ms
draft   = 3 × 1.2 = 3.6 ms
```

| h | experts | step_ms |
|---|---|---|
| 0.90 | 4.55 | **26.5** |
| 0.44 | 25.48 | **47.4** |

Rapporto atteso: `47.4 / 26.5 = 1.79×`.

Empirico, stesso modello e stessa macchina, due checkout:

| Run | h | token | ms | ms/token |
|---|---|---|---|---|
| engine 0.1.27 | 86.0% | 31047 | 398471 | **12.83** |
| engine 0.1.30 | 43.8% | 1420 | 36530 | **25.72** |

```
25.72 / 12.83 = 2.00×  più lento
```

Lo scarto 1.79 vs 2.00 è il drain non lineare della pool CPU: con 308 miss/token la coda si somma e la window aspetta il worker più lento, non la media.

### Costo reale di un miss, ricavato dai log

```
lookups/token:  779114 / 1420   = 548.7   (h = 44%)
                18414313 / 31047 = 593.1  (h = 86%)
                atteso ~480 = 48 layer × 10 expert

miss/token:     548.7 × 0.562 = 308.4
                593.1 × 0.140 =  83.0      → Δ = 225.4

Δms/token:      25.72 − 12.83 = 12.89

per miss = 12.89 / 225.4 = 0.057 ms = 57 µs
```

Coerenza: 480 expert tutti miss → 480 × 57 µs = 27.4 ms, a fronte dei `cpu_all_miss_ms = 15.8` del CostModel. Il modello sottostima ~1.7× perché assume i 19 worker perfettamente parallelizzati.

### Catena completa

```
9615 slot identici
   → ordine diverso
   → massa coperta 39.1% vs 93.0%
   → hit rate 44% vs 90%
   → miss 308 vs 83 /token
   → 25.7 vs 12.8 ms/token
   → ~45 vs ~90 tok/s
```

Nessun bug, nessuna regressione di codice.

---

## 2. Creare un profilo by-frequency da zero

### 2.1 Tokenizza un testo del tuo workload reale

```bash
cd /home/albus/workspaces/AI/Strata/RiStarta/Strata

python3 - <<'PY'
import sys; sys.path.insert(0, "tools")
from strata_tokenizer import Tokenizer

gguf = "/home/albus/workspaces/rag_rss/models/llm_models/models/IQ3_S/" \
       "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
tk   = Tokenizer.from_gguf(gguf)
text = open("/tmp/corpus.txt", encoding="utf-8").read()
ids  = tk.encode(text, parse_special=True)
open("/tmp/tokens.txt", "w").write(" ".join(map(str, ids)))
print(len(ids), "tokens")
PY
```

`/tmp/corpus.txt` = incolla codice, chat, tool call: quello che il modello fa davvero. Il profilo è una fotografia della distribuzione di prompt. Se il corpus non la rappresenta, l'hit rate non sale.

### 2.2 Run one-shot che scarica la trace

```bash
engine/strata \
  --pack /home/albus/workspaces/AI/Strata-data/packs/iq3_s \
  --native /home/albus/workspaces/rag_rss/models/llm_models/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf /home/albus/workspaces/rag_rss/models/llm_models/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --tokens-file /tmp/tokens.txt \
  --max-new 12000 \
  --max-context 32768 \
  --dump-routing /tmp/trace.bin \
  --expert-cache 0 --prefill 0 --spec 0
```

Perché proprio quei flag:

| Flag | Motivo nel codice |
|---|---|
| `--prefill 0` | **il più importante**. Col prefill attivo il prompt viaggia in batch dentro `prefill.run()` e non passa da `drive_pool`: zero record per i token del prompt. La trace la scrivono solo `drive_pool` ([`generate.cpp:590`](../../src/program/generate.cpp:590)) e `drive_pool_multi` ([`generate.cpp:622`](../../src/program/generate.cpp:622)) |
| `--expert-cache 0` | disattiva la cache → `hit_fn = nullptr` ([`generate.cpp:2789`](../../src/program/generate.cpp:2789)). La pool vede tutti i k expert senza ambiguità hit/miss |
| `--spec 0` | usa `drive_pool` invece di `drive_pool_multi`. Il multi scrive **pesi unitari** ([`generate.cpp:627`](../../src/program/generate.cpp:627)), il single scrive i pesi reali del router |
| nessun `--expert-profile` | la trace non deve essere contaminata da una residenza precedente |
| **non** `--no-pool` | vietato: [`generate.cpp:2775`](../../src/program/generate.cpp:2775) esce con "needs the expert pool" |

Serve la **GPU libera** (~18 GiB). Alla fine stampa:

```
routing dumped         /tmp/trace.bin (N records of layer, k, ids, weights)
```

Formato record: `int32 layer, int32 k, k int32 ids, k float weights`. Dimensione attesa:

```
(8 + 8×10) byte × 48 layer × ~13000 token ≈ 66 MB
```

### 2.3 Costruisci il profilo

```bash
mkdir -p /home/albus/workspaces/AI/Strata-data/profiles

python3 tools/make_profile.py /tmp/trace.bin --no-base \
  --out /home/albus/workspaces/AI/Strata-data/profiles/iq3_s-chat.bin
```

`--no-base` è la chiave per "da capo": senza, il tool mette prima il ranking del `--base` e le tue coppie finiscono dietro ([`make_profile.py:82`](../../tools/make_profile.py:82)). Con `--no-base` l'ordine è: coppie dalla trace per frequenza decrescente, poi le mancanti **interleaved tra le layer** ([`make_profile.py:91`](../../tools/make_profile.py:91)) — esattamente ciò che evita la trappola layer-major.

Più trace si fondono sommando le frequenze:

```bash
python3 tools/make_profile.py /tmp/trace-code.bin /tmp/trace-chat.bin /tmp/trace-tools.bin \
  --no-base --out /home/albus/workspaces/AI/Strata-data/profiles/iq3_s-mixed.bin
```

Modello potato (Coder: 256 expert su 512):

```bash
python3 tools/make_profile.py /tmp/trace.bin --no-base --n-expert 256 \
  --out /home/albus/workspaces/AI/Strata-data/profiles/coder-iq1_m.bin
```

### 2.4 Punta la config al profilo e misura

Nella config `strata-<modello>.json`, campo `args`:

```json
"--expert-profile", "/home/albus/workspaces/AI/Strata-data/profiles/iq3_s-chat.bin",
"--expert-cache", "auto",
"--expert-cache-per-layer"
```

Poi nel log cerchi:

```
profile ...: 24576 ranked pairs, built for 24576 slots
expert cache 9615 slots, 17.29 GiB of VRAM
pre-filled 9615 of 9615 slots from the profile; slot 0 verified
decode expert cache hit rate: 8x.x%
```

### 2.5 Quanta trace serve

Per layer hai `10 × N_token` campioni su 512 expert:

```
per rankare il top-200 (quelli che entrano nei 9615/48 slot):
    10N ≫ 200 · ln(512) ≈ 6200   →  N ≳ 600
per la coda (512 expert):
    10N / 512 ≥ 20               →  N ≥ 1024
consigliato:
    N = 10000 – 20000            →  100k–200k campioni/layer, ~200–400 per expert
```

Sotto ~2000 token di decode il ranking della coda è rumore; sopra ~30000 non guadagni nulla.

### 2.6 Portabilità

| Caso | Funziona? |
|---|---|
| Stesso modello, altra quantizzazione (Q2_0 ↔ IQ3_S) | **sì**, riusa lo stesso file. Il routing dipende dai pesi, non da come sono quantizzati |
| Modello potato (256 expert) | **sì**, con `--n-expert 256` |
| Architettura con N layer o N expert diverso | **no**: l'engine rifiuta il file. Serve trace nuova e il tool parametrizzato (`N_LAYER, N_EXPERT` sono hard-coded in [`make_profile.py:23`](../../tools/make_profile.py:23)) |
| Modello denso (Qwen3.8-27B) | **irrilevante**: nessun expert, nessun profilo |
| Workload diverso (code → chat) | il profilo resta valido ma rende meno: rigeneralo |

Tieni i profili **fuori dal repo** (sono in `.gitignore`):

```
Strata-data/profiles/qwen3.8-flash-next.bin
Strata-data/profiles/coder-iq1_m.bin
```

`--expert-profile` è anche obbligatorio per due funzioni che altrimenti non partono: layer split su più GPU ([`generate.cpp:1264`](../../src/program/generate.cpp:1264)) e `--resident-cpu-experts` ([`generate.cpp:1250`](../../src/program/generate.cpp:1250)).

### 2.7 Controllo di cosa resta residente

| Flag | Effetto |
|---|---|
| `--expert-cache auto` | dimensiona dagli slot che la VRAM libera paga (meno reserve, buffer prompt, drafter) |
| `--expert-cache N` | tronca la lista a N: prova "cosa dà 2000 slot" senza rigenerare il file |
| `--expert-cache 0` | cache disattivata, tutto sulla CPU pool |
| `--expert-cache-per-layer` | ogni layer ha i **suoi** slot invece di un counter condiviso |

L'ultimo è decisivo. Dal help in [`generate.cpp:506`](../../src/program/generate.cpp:506): senza per-layer il counter condiviso manda i primi slot tutti alle layer basse → **2.97%** di hit. Con per-layer: **21.4%** a 8 slot/layer, **70.4%** a 64/layer.

Il profilo è solo lo **stato iniziale**: l'adaptive tier scambia poi gli expert in base a cosa la conversazione instrada realmente, ogni `--adapt-every` round.

---

## 3. Qwen3.8-27B (dense) da riga di comando

### 3.1 Build

```bash
cmake -B build-dense -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_TESTS=OFF
cmake --build build-dense --target xolotl-dense -j
```

Scorciatoia che fa tutto (download GGUF, tokenizer, build, config):

```bash
python3 tools/dense_setup.py --gguf /path/Qwen3.8-27B-UD-Q4_K_M.gguf --build --arch 86 --context 32768
# oppure scarica lui il modello
python3 tools/dense_setup.py --download UD-Q4_K_M --build --arch 86
```

Nessun pack step: `strata-dense` legge i blocchi GGUF come sono, con qualsiasi tipo i GEMV nativi accettano (Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, IQ3_S, IQ4_XS, IQ2_*).

Verifica dei kernel senza modello:

```bash
build-dense/strata-dense --selftest
```

### 3.2 Flag disponibili

Da [`src/program/dense_main.cpp:254`](../../src/program/dense_main.cpp:254) — sono pochi, e `--serve` è **obbligatorio**:

```
strata-dense --serve --native <model.gguf> [--context N]
```

| Flag | Note |
|---|---|
| `--serve` | obbligatorio. Non esiste una modalità one-shot: il programma parla sempre il protocollo su stdin/stdout |
| `--native` / `--model` / `--gguf` | path del GGUF (singolo file o primo shard di uno split) |
| `--context` / `-c` | celle di contesto, default 32768 |
| `--selftest` | testa i kernel CUDA contro una reference CPU, non serve il modello |

I flag che `setup` passa a `strata` (`--gpu-layers`, `--cache`, …) qui sono **ignorati senza errore** ([`dense_main.cpp:258`](../../src/program/dense_main.cpp:258)).

### 3.3 Protocollo stdin/stdout

Documentato in [`src/program/dense_main.cpp:1`](../../src/program/dense_main.cpp:1):

```
out   INFO key=value ...                    fatti per il Monitor
out   READY <context> stop                  l'engine è pronto
in    GEN <max_new> [key=value ...] id,id,...
        chiavi: temperature top_p top_k min_p
                penalty_last_n penalty_repeat penalty_freq penalty_present seed
out   PP <posizione> <prompt token> <ms> <tok/s>       ogni ~32 token di prompt
out   T <id>                                          un token generato
out   DONE <generati> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel> 0 0 <reuse>
in    STOP | QUIT
```

### 3.4 Run manuale

```bash
GGUF=/media/albus/windows/Users/mio/models/llm_models/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

# tokenizza la domanda
python3 - <<'PY' > /tmp/q.txt
import sys; sys.path.insert(0, "tools")
from strata_tokenizer import Tokenizer
tk = Tokenizer.from_gguf("/media/albus/windows/Users/mio/models/llm_models/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf")
print(",".join(map(str, tk.encode("Spiega in due frasi cosa fa una GPU.", parse_special=True))))
PY

# passa il GEN sul stdin, leggi i token
printf 'GEN 128 temperature=0 top_k=1 %s\nQUIT\n' "$(cat /tmp/q.txt)" \
  | build-dense/strata-dense --serve --native "$GGUF" --context 4096
```

`temperature=0` = greedy ([`include/strata/kernels/sampler.hpp:19`](../../include/strata/kernels/sampler.hpp:19)).

### 3.5 Run tramite server (OpenAI/Anthropic + web app)

```bash
python3 serve/server.py --engine strata --config strata-qwen3.8-27b.json --port 8080
```

Config attuale in [`strata-qwen3.8-27b.json`](strata-qwen3.8-27b.json): `exe` punta a `build-dense/strata-dense`, `tokenizer` a `data/packs/qwen3.8-27b/tokenizer`. Chat da terminale:

```bash
python3 chat.py --port 8080
```

### 3.6 Budget VRAM e pesi in host memory

Il modello calcola da solo quanto sta in VRAM ([`src/core/dense_model.cpp:441`](../../src/core/dense_model.cpp:441)):

```
budget = VRAM libera − KV cache − 1 GiB di margine
```

Quello che non sta in VRAM finisce in **pinned host memory mappata**, e il GEMV lo rilegge via PCIe a ogni token. Il log lo dice esplicitamente:

```
strata-dense: 14 tensors (11.20 GiB of 15.40) live in host memory and are read over PCIe
```

Quella riga è il collo di bottiglia numero uno del percorso denso. Per controllarla:

| Leva | Effetto |
|---|---|
| `--context N` più corto | la KV cache occupa meno, più pesi entrano in VRAM |
| `STRATA_DENSE_GPU_MIB=<MiB>` | budget esplicito per i pesi, ignora l'auto-calcolo ([`dense_model.cpp:451`](../../src/core/dense_model.cpp:451)) |
| quantizzazione più piccola | meno byte totali da piazzare |

Se la KV non sta nemmeno con il margine, l'engine si ferma con:

```
dense model: the KV cache for a context of N needs X MiB, but only Y MiB of VRAM are free
(1 GiB is kept as margin). Use a shorter --context.
```

### 3.7 Debug

```bash
STRATA_DENSE_DEBUG=0:2 build-dense/strata-dense --serve --native "$GGUF"
```

Traccia le statistiche dei vettori device per i passi con `0 <= posizione < 2`, layer per layer ([`dense_model.cpp:417`](../../src/core/dense_model.cpp:417), [`:286`](../../src/core/dense_model.cpp:286)). Utile per capire se un layer produce NaN o scala sbagliata.

### 3.8 Limiti noti del percorso denso

- **Nessuna speculative decoding.** Il loop è autoregressive puro, un token per forward pass. Il GDN ha uno stato ricorrente che avanza un token alla volta e il residual stream crea dipendenza sequenziale: non si possono batchare T token in un solo pass.
- **Prompt letto un token alla volta** ([`dense_main.cpp:20`](../../src/program/dense_main.cpp:20)). Corretto e confrontabile con llama.cpp; un prefill batchato è l'ottimizzazione successiva, non un prerequisito.
- **Conversation cache solo per prefissi.** Lo stato GDN non si può rollbaccare: una richiesta riusa lo stato solo se tutto ciò che ha consumato è prefisso del nuovo prompt. Altrimenti reset e prompt riletto ([`dense_main.cpp:16`](../../src/program/dense_main.cpp:16)).

---

## 4. MoE (Qwen3.8-Flash-Next) — riferimento rapido

```bash
# one-shot, senza server
engine/strata \
  --pack /home/albus/workspaces/AI/Strata-data/packs/iq3_s \
  --native <modello>-00001-of-00002.gguf \
  --ple-gguf <modello>-00002-of-00002.gguf \
  --expert-profile /home/albus/workspaces/AI/Strata-data/profiles/iq3_s-chat.bin \
  --expert-cache auto --expert-cache-per-layer \
  --prefill auto --spec 4 --spec-min-p 0.5 \
  --mtp /home/albus/workspaces/AI/Strata-data/mtp/rt \
  --max-context 128000 --kv int8 --kv-resident 32768 \
  --tokens-file /tmp/tokens.txt --max-new 512
```

```bash
# server
python3 serve/server.py --engine strata --config strata-iq3_s.json --port 4332
```

Speculazione: `--spec T` (T ≤ 8) abilita la verify window, `--mtp <dir>` il draft layer, `--suffix-draft N` il prompt lookup (default 3, `0` lo spegne). Il drafter suffix non ha pesi e costa microsecondi: paga su codice e testo ripetuto, +6-11% ([`generate.cpp:1358`](../../src/program/generate.cpp:1358)).

Diagnostica in ogni run:

```
session is up (engine 0.1.30)
expert cache 9615 slots, 17.29 GiB of VRAM
decode expert cache hit rate: 8x.x% (N hits / M lookups)
speculation   N rounds of 4, drafts accepted X of Y (0.65), 3.1 tokens per round
```

Se l'hit rate scende sotto ~80%, il sospetto numero uno è il profilo — non il codice dell'engine.
