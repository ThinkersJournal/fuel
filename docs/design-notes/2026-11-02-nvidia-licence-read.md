# Nvidia terms that apply to what Fuel builds on: a read of the published texts

Board 307, PM tasks A and A2 (2026-11-01, relayed from CireSnave: "Have the lane check each Nvidia product we use").
Written 2026-11-02 by Fuel 1 (Claude Sonnet 5.5). **This is a reading of published text. It is NOT legal advice,
it reaches NO verdict on infringement, validity or licensing, and nobody at Nvidia was contacted.** Every
source not read is marked **UNREAD**. Quotes are copied from the raw fetched text (files hashed below), not from
a summary.

## 1. Method and provenance

- Raw files were fetched with `curl` on 2026-11-02 (UTC) and the quoted sentences were located with a text search over
  the raw text (HTML tags removed for the web pages). Two sub-agents fetched the same pages first through a summarising
  fetch tool; their fragments were NOT used where this note quotes: each quote below was re-found in the raw text.
- GitHub files are pinned to a commit, so the text can be re-fetched:

| Source | Pinned to | sha256 (first 16) of the fetched file |
|---|---|---|
| TensorRT-LLM `LICENSE` | `NVIDIA/TensorRT-LLM` commit `cf96e9ec516a9c6f8e04de649926f7598b90fe94` (branch main) | `a56e691155bb532e` |
| CUTLASS `LICENSE.txt` | `NVIDIA/cutlass` commit `e13f5cd4142c43efc37c2f6339923eb4a57afb99` (branch main) | `ad1513ca983da737` |
| NCCL `LICENSE.txt` | `NVIDIA/nccl` commit `3e46c089f4ae4732722a419eb6096aead7e47a30` (branch master) | `c1f53beabe4dbf05` |

- Web pages (no commit to pin; version and date as printed on the page): CUDA Toolkit EULA
  `https://docs.nvidia.com/cuda/eula/index.html` ("v13.4", "Last updated: January 26, 2026"); cuDNN licence
  `https://docs.nvidia.com/cudnn/reference/eula.md` (licence text dated "(v. February 22, 2022)"); CUTLASS DSL
  licence `https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/license.html` ("(v. May 8, 2025)"; an earlier
  `4.4.1` copy at the same path was also fetched and also contains no occurrence of "patent"); cuBLAS documentation
  `https://docs.nvidia.com/cuda/cublas/index.html` (page footer "Last updated on Oct 12, 2026").

## 2. What Fuel uses (from `Cargo.lock` and manifests at `9dc33d611`)

`Cargo.lock` lists these Nvidia-facing crates, all published by Baracuda at `0.1.0-alpha.80`: `baracuda-cuda-sys`,
`-driver`, `-core`, `-types` (CUDA driver and runtime bindings), `-nvrtc`/`-nvrtc-sys`, `-cublas`/`-cublas-sys`,
`-cudnn`/`-cudnn-sys`, `-curand`/`-curand-sys`, `-nccl`/`-nccl-sys`, `-cutlass`/`-cutlass-sys`/
`-cutlass-kernels-sys`, `-kernels-sys`, `-kernels-types`, `-cuda-seam`, `-forge`, `-cuda-emit`. These are Rust bindings
and build scripts; the Nvidia libraries themselves come from the user's installed CUDA Toolkit / cuDNN / NCCL at run
time. **No TensorRT-LLM** appears in the lockfile or in any manifest. `baracuda-nvml` is not in the lockfile. `vulkane`
talks to Vulkan (a Khronos API), not an Nvidia product. The `baracuda-cutlass-kernels-sys` `NOTICE` (crate
`0.1.0-alpha.80`, read from the downloaded crate source) says: "The CUTLASS template instantiations under `kernels/`
derive from NVIDIA's CUTLASS examples (https://github.com/NVIDIA/cutlass), licensed under BSD 3-Clause. CUTLASS itself
is not redistributed — its headers are fetched at build time by `baracuda-cutlass-sys`."

## 3. Per product

Patent wording, quoted, with the grant question (A2 item 1), then use and redistribution (A2 item 2 and A item "commercial
open-source project"), whether it embodies scaled low-bit dot kernels (A2 item 3), and software or hardware (A2 item 4).

### CUDA Toolkit EULA v13.4 (CUDA runtime, NVRTC, cuBLAS, cuBLASLt)

- Patent wording in Nvidia's own text: Section 2.4: "NVIDIA does not grant to you under this Agreement any necessary
  patent or other rights with respect to any audio and/or video encoders and decoders." The warranty disclaimer also
  contains "OR THE ABSENCE OF ANY DEFECTS THEREIN, WHETHER LATENT OR PATENT."
- The agreement also reproduces the Apache License 2.0, introduced by the sentence "Licensee’s use of the Thrust library is
  subject to the terms and conditions of the Apache License Version 2.0." That text says "3. Grant of Patent License.
  Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide,
  non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license". It is a
  grant by that component's contributors, not by Nvidia's agreement; which other attachment entries use Apache text:
  **UNREAD**.
- The main grant (Section 1.1.1) reads: "NVIDIA hereby grants you a non-exclusive, non-transferable license, without the
  right to sublicense (except as expressly provided in this Agreement) to: Install and use the SDK, Modify and create
  derivative works of sample source code delivered in the SDK, and ...". It does not mention patents either way.
- Distribution: "Your application must have material additional functionality, beyond the included portions of the SDK.
  The distributable portions of the SDK shall only be accessed by your application." and "Unless a developer tool is
  identified in this Agreement as distributable, it is delivered for your internal use only." The attachment listing
  distributable files says "may be distributed with applications developed by you" (the fetched summary named the
  cudart, cuBLAS/cuBLASLt and NVRTC runtime files; the file names were not re-read in the raw text: **UNREAD**).
- A sentence in Nvidia's documentation inviting commercial use: none found in the EULA (the word "commercial" appears only
  in pre-release and US-government clauses). The cuBLAS page has zero occurrences of "commercial". **UNREAD** for
  other Nvidia pages.
- cuBLAS documentation (raw page text) describes scaled low-bit modes: a heading "Outer Vector Scaling for FP8 Data
  Types", whose text begins "This scaling mode (also known as channelwise or rowwise scaling) is a refinement of the
  tensorwide scaling", block-scaling sections for FP8 and FP4 ("Block Scaling for FP8 and FP4 Data Types", 13
  occurrences of "Block Scaling"), and for emulation "Elements in the same row of the A matrix, or column of the B
  matrix, share the same scaling factor". That is documentation evidence that cuBLAS embodies per-vector or per-block
  scaled low-bit matrix multiplies for FP8/FP4. Whether it has int8 per-vector kernels: **UNREAD** (section bodies not
  read).
- The cuBLAS page carries the usual document notice: "No license, either expressed or implied, is granted under any
  NVIDIA patent right, copyright, or other NVIDIA intellectual property right under this document." and "Use of such
  information may require a license from a third party under the patents or other intellectual property rights of the
  third party, or a license from NVIDIA under the patents or other intellectual property rights of NVIDIA."
- Software (libraries run on Nvidia GPUs).

### cuDNN (licence text dated "(v. February 22, 2022)")

- The only occurrence of "patent" is the warranty disclaimer "WHETHER LATENT OR PATENT." No patent grant sentence.
- Grant: "NVIDIA hereby grants you a non-exclusive, nontransferable license, without the right to sublicense (except as
  expressly provided in this Agreement) to: 1. Install and use the SDK, 2. Modify and create derivative works of sample
  ...". Distribution: "the following portions of the SDK are distributable under the Agreement: the runtime files .so
  and .dll", with the same "material additional functionality" condition as above.
- Commercial-use invitation: none found. Scaled low-bit kernels: **UNREAD** (cuDNN documentation was not read).
- Software.

### CUTLASS C++ (BSD-3-Clause)

- `LICENSE.txt` begins "Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved." and
  "SPDX-License-Identifier: BSD-3-Clause". The word "patent" does not occur in it: the text is silent on patents.
- "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
  following conditions are met:" (keep the notice in source; reproduce it in binary form; no endorsement).
- Per the NOTICE of the Baracuda crate (section 2), Fuel's build downloads these headers and compiles kernels derived
  from CUTLASS examples. The repository `NOTICE` of CUTLASS: **UNREAD**.
- Embodies scaled low-bit kernels: **UNREAD** (the headers were not browsed). Software.

### CUTLASS Python DSL / CuTe DSL ("Software License Agreement", "(v. May 8, 2025)")

Not used by Fuel (not in the lockfile or build scripts). Read because the task named it. No occurrence of "patent" in
the text. Terms of interest for anyone who does adopt it: the grant is "a limited, non-exclusive, revocable,
non-transferable, and non-sublicensable (except as expressly granted in this Agreement) license"; "and the developed
applications are only for use in systems with NVIDIA GPUs"; Section 1.2 gives Nvidia "a non-exclusive, perpetual,
irrevocable, sublicensable, worldwide, royalty-free, fully paid-up and transferable license, under your intellectual
property rights, to ... sell, offer for sale ... any Derivatives created by or for you"; Section 2.10 forbids
"Replace any NVIDIA software components in the Software that are governed by this Agreement with other software that
implements NVIDIA APIs"; Section 9.1 ends the licence if you "commence or participate in any legal proceeding against
NVIDIA with respect to the Software". The CuTe DSL samples on GitHub are under a different licence per a search
snippet: **UNREAD**.

### NCCL (Apache 2.0 plus a BSD part)

- `LICENSE.txt` begins "Most of this project is licensed under the Apache-2 license (right below)." The patent grant is
  Apache 2.0 Section 3: "Subject to the terms and conditions of this License, each Contributor hereby grants to You a
  perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent
  license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license
  applies only to those patent claims licensable by such Contributor that are necessarily infringed by their
  Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was
  submitted. If You institute patent litigation against any entity ... then any patent licenses granted to You under
  this License for that Work shall terminate as of the date such litigation is filed."
- Redistribution is in Section 4 of the same text. NCCL is a communication library; it has no dot-product kernels
  (its tree was not read: **UNREAD**). Software.

### TensorRT-LLM (Apache 2.0, with bundled third-party texts)

Not used by Fuel. `LICENSE` says "This project is licensed under the Apache 2.0 license, whose full license text is
available below." and contains the same Apache Section 3 patent grant (the quoted sentences are at lines 226 to 240 of
the pinned file). The same file also bundles a separate "LTX-2 Community License Agreement". Its scope sentence is
"The LTX-2 Community License Agreement applies only to use of the code in the
`tensorrt_llm/_torch/visual_gen/models/ltx2/` directory of this project." and its text says "Entities with annual
revenues of at least $10,000,000 (the "Commercial Entities") are required to obtain a paid commercial" licence. A
`NOTICE` file: none at the pinned commit's root (the raw path returned HTTP 404; the directory listing was read by a
sub-agent only). Whether it embodies scaled low-bit kernels: **UNREAD**. Software.

## 4. CPU

Nvidia software on this list is GPU software. Whether Nvidia ships any CPU per-vector-scaled dot-product code:
**UNREAD** (no CPU product was found in the lists above; absence of a finding is not a finding of absence).

## 5. Dates before 2020-08-28 (cite only, no opinion)

Dates are those printed on the fetched pages (read by a sub-agent through a summarising tool, so re-check before
citing; the underlying documents were not read in full):

- Drumond, Lin, Jaggi, Falsafi, "Training DNNs with Hybrid Block Floating Point", arXiv 1804.01526, v1 dated
  4 Apr 2018: dot products in block floating point (a shared exponent per block).
- Köster et al., "Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks", arXiv
  1711.02213, v1 dated 6 Nov 2017: tensors share an exponent (granularity is the tensor, not the vector).
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", arXiv
  1712.05877, submitted 15 Dec 2017: integer-only inference (scale granularity **UNREAD**).
- US 10,579,334 B2, Lo and Chung (Microsoft), "Block floating point computations using shared exponents": filed
  2018-05-08, published as US20190347072A1 on 2019-11-14, granted 2020-03-03: mantissas grouped with a shared
  exponent, dot products on the mantissas in integer arithmetic.
- After the date line (listed so nobody cites them as earlier): VS-Quant, arXiv 2102.04503, submitted 8 Feb 2021;
  Microsoft Floating Point, NeurIPS 2020 (listed 2020-12-01, page **UNREAD**).
- **UNREAD / not searched:** GGML and llama.cpp block-quantized dot products, Intel and ARM documents, other
  conference papers.

## 6. What this note does not say

Nothing here says Fuel needs, lacks or can obtain a licence. It does not interpret any clause, say whether hardware
or software sale exhausts a patent, or say whether using a vendor library is covered by a vendor's patents. Those are
questions for counsel. The list of UNREAD items above is the work remaining to make this note complete.
