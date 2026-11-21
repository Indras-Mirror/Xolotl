// src/program/inkling_vision.h - Inkling-2-Mini's image encoder inside xolotl-inkling (the inkling2v tower of
// llama.cpp PR #32694, tools/mtmd/models/inkling2v.cpp, re-built on ggml-backend; MIT).
//
// The weights come from the model's mmproj GGUF (`convert_hf_to_gguf.py <hf> --mmproj`, projector_type inkling2v,
// ~0.4B params).  `encode()` turns an image file into the SVE1 record the server already hands the engine: int32
// {'SVE1', n, nx, ny, n_embd} + float32 [n][n_embd], the rows that replace the prompt's `<|unused_200054|>` image
// tokens (n = (width/40) * (height/40), 2x2-merged 20-px patches, each row in the text model's 2304-d space).
//
//     xolotl-inkling --encode <image> <out.sve1> --mmproj <mmproj.gguf> [--vision-device cpu|gpu]
//
// Weights are loaded into host memory once and uploaded to the GPU per picture (640 MB, ~0.1 s), so a text-only
// session never pays for them in VRAM.
#pragma once

#include "ggml-backend.h"

#include <string>

namespace inkling_vision {

struct Encoder;

// Load an mmproj (clip GGUF with clip.vision.projector_type = inkling2v).  `gpu` may be null (CPU only); with
// `use_gpu` false the encoder always runs on the CPU.  Returns null and sets `err` on failure.
Encoder* load(const std::string& mmproj, ggml_backend_t gpu, ggml_backend_t cpu, bool use_gpu, std::string& err);

// Encode `image` (PNG/JPEG/BMP/GIF, as stb_image reads them) into `out` as one SVE1 record.  `max_tokens` bounds
// the rows per picture (0 = 4096, mtmd's inkling2v cap): a bigger picture is resized down, aspect kept, 40-px
// aligned.  Returns the row count, or 0 with `err` set.
int encode(Encoder& enc, const std::string& image, const std::string& out, int max_tokens, std::string& err);

void free(Encoder* enc);

}   // namespace inkling_vision
