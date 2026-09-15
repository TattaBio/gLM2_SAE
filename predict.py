#!/usr/bin/env python3
"""Call SAE features on intergenic sequence, straight from a DNA FASTA.

    python predict.py contigs.fasta -o features.csv

Pipeline: prodigal gene calling (pyrodigal) -> CDS/IGS elements in contig order
-> gLM2-650M layer-24 residuals -> SAE over the intergenic positions
-> one CSV row per feature run, in contig coordinates.

A "run" is a stretch of intergenic positions where one latent keeps firing:
scattered single-position hits are noise, so a latent must fire on at least
`--min-run` positions spanning at least that many bp (tolerating gaps of up to
`--run-gap` bp), and latents firing on more than `--max-density` of a window's
intergenic positions are dropped as ubiquitous.
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass

import numpy as np
import torch

from elements import Coord, build_contig, gene_finder, read_fasta, tile, tokenize_and_align
from sae import CONTEXT_SIZE, SAE_LAYER, SAE_REPO, JumpReluSae, load_glm2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIELDS = [
    "contig_id",
    "feature_id",
    "start",
    "end",
    "length",
    "n_active",
    "max_activation",
    "total_activation",
    "left_gene",
    "right_gene",
]
GENE_FIELDS = ["contig_id", "gene_index", "start", "end", "strand", "n_aa"]


@dataclass
class Window:
    contig_id: str
    n_genes: int
    token_ids: list[int]
    coords: list[Coord]


def _runs(feat: np.ndarray, pos: np.ndarray, min_run: int, run_gap: int):
    """Cut latent-major hits into runs, keeping those spanning >= min_run bp.

    `feat`/`pos` are parallel hit arrays sorted by latent then position. A run
    breaks at a new latent or a gap wider than `run_gap`. Returns the
    `[start, stop)` bounds of the survivors.
    """
    breaks = np.empty(len(feat), dtype=bool)
    breaks[0] = True
    np.logical_or(feat[1:] != feat[:-1], pos[1:] - pos[:-1] - 1 > run_gap, out=breaks[1:])
    start = np.flatnonzero(breaks)
    stop = np.append(start[1:], len(feat))
    survives = pos[stop - 1] - pos[start] + 1 >= min_run
    return start[survives], stop[survives]


def window_rows(acts: torch.Tensor, window: Window, igs_coords: list[Coord], args) -> list[dict]:
    """One window's (n_igs, d_sae) activations -> feature-run rows."""
    fired = acts > 0
    n_fire = fired.sum(0)
    keep = (n_fire >= args.min_run) & (n_fire <= args.max_density * acts.shape[0])
    # Filtering on device keeps only the surviving hits (~L0 per position) on
    # the host, not the dense activation matrix.
    feat_t, row_t = (fired & keep).t().nonzero(as_tuple=True)
    if feat_t.numel() == 0:
        return []
    feat = feat_t.cpu().numpy()
    vals = acts[row_t, feat_t].cpu().numpy()
    rows_np = row_t.cpu().numpy()
    pos = np.array([igs_coords[r][0] for r in rows_np], dtype=np.int64)

    rows = []
    for start, stop in zip(*_runs(feat, pos, args.min_run, args.run_gap), strict=True):
        run_pos, run_vals = pos[start:stop], vals[start:stop]
        left_gene = igs_coords[rows_np[start]][1]
        rows.append(
            {
                "contig_id": window.contig_id,
                "feature_id": int(feat[start]),
                "start": int(run_pos[0]),
                "end": int(run_pos[-1]),
                "length": int(run_pos[-1] - run_pos[0] + 1),
                "n_active": len(run_pos),
                "max_activation": round(float(run_vals.max()), 4),
                "total_activation": round(float(run_vals.sum()), 4),
                "left_gene": left_gene + 1 if left_gene >= 0 else "",
                "right_gene": left_gene + 2 if left_gene + 1 < window.n_genes else "",
            }
        )
    return rows


def encode_batch(windows: list[Window], model, tokenizer, sae, device, args) -> list[dict]:
    """Run a padded batch of windows through gLM2 + the SAE."""
    batch = tokenizer.pad({"input_ids": [w.token_ids for w in windows]}, return_tensors="pt")
    # The mask gets expanded to (batch, heads, seq, seq) inside attention, the
    # single biggest allocation of the forward pass -- skip it when nothing was
    # padded.
    attn = None if len(set(len(w.token_ids) for w in windows)) == 1 else batch["attention_mask"].bool().to(device)

    dev_type = device.split(":")[0]
    with torch.inference_mode(), torch.autocast(dev_type, torch.bfloat16, enabled=dev_type == "cuda"):
        # return_dict is explicit because falling back to the config is deprecated.
        resid = model(batch["input_ids"].to(device), attention_mask=attn, return_dict=True).last_hidden_state

    rows = []
    for i, window in enumerate(windows):
        # Coding tokens are part of the forward pass -- they are the context
        # gLM2 reads the intergenic regions in -- but only intergenic residuals
        # carry a coordinate, and only those are encoded. Ambiguous bases
        # tokenize to <unk>, which the SAE never saw in training.
        idx = [
            j for j, c in enumerate(window.coords) if c is not None and window.token_ids[j] != tokenizer.unk_token_id
        ]
        if not idx:
            continue
        acts = sae.encode(resid[i, torch.as_tensor(idx, device=device)].float())
        window_out = window_rows(acts, window, [window.coords[j] for j in idx], args)
        window_out.sort(key=lambda r: (r["start"], r["feature_id"]))
        rows.extend(window_out)
    return rows


def contigs_from_fasta(path: str):
    finder = gene_finder()
    empty = True
    for name, seq in read_fasta(path):
        empty = False
        contig = build_contig(name, seq, finder.find_genes(seq))
        igs_bp = sum(len(e.body) for e in contig.elements if e.kind == "IGS")
        logger.info("%s: %d bp, %d genes, %d intergenic bp", name, len(seq), len(contig.genes), igs_bp)
        yield contig
    if empty:
        sys.exit(f"No sequences found in {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fasta", help="DNA FASTA (one or more contigs).")
    p.add_argument("-o", "--output", default="sae_features.csv", help="Output CSV of feature runs.")
    p.add_argument("-g", "--genes-output", help="Output CSV of the prodigal gene calls, written alongside.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda, cpu or mps.")
    p.add_argument("--batch-windows", type=int, default=8, help="Windows per forward pass.")
    p.add_argument("--context-size", type=int, default=CONTEXT_SIZE, help="Tokens per window.")
    p.add_argument("--min-run", type=int, default=8, help="Minimum firing positions and bp span per run.")
    p.add_argument("--run-gap", type=int, default=3, help="Non-firing bp tolerated inside a run.")
    p.add_argument("--max-density", type=float, default=0.5, help="Drop latents firing on more than this fraction.")
    args = p.parse_args()

    logger.info("loading gLM2 (layer %d) and SAE %s on %s", SAE_LAYER, SAE_REPO, args.device)
    model, tokenizer = load_glm2(args.device)
    sae = JumpReluSae(device=args.device)

    out = open(args.output, "w", newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    genes_out = genes_writer = None
    if args.genes_output:
        genes_out = open(args.genes_output, "w", newline="")
        genes_writer = csv.DictWriter(genes_out, fieldnames=GENE_FIELDS)
        genes_writer.writeheader()

    pending: list[Window] = []
    n_rows = 0

    def flush() -> None:
        nonlocal n_rows
        if not pending:
            return
        rows = encode_batch(pending, model, tokenizer, sae, args.device, args)
        writer.writerows(rows)
        out.flush()
        n_rows += len(rows)
        pending.clear()

    for contig in contigs_from_fasta(args.fasta):
        if genes_writer:
            genes_writer.writerows(
                {
                    "contig_id": contig.id,
                    "gene_index": i + 1,
                    "start": g.start,
                    "end": g.end,
                    "strand": g.strand,
                    "n_aa": len(g.body),
                }
                for i, g in enumerate(contig.genes)
            )
        token_ids, coords = tokenize_and_align(contig, tokenizer)
        for window_tokens, window_coords in tile(token_ids, coords, args.context_size):
            pending.append(Window(contig.id, len(contig.genes), window_tokens, window_coords))
            if len(pending) >= args.batch_windows:
                flush()
    flush()

    out.close()
    if genes_out:
        genes_out.close()
    logger.info("wrote %d feature runs to %s", n_rows, args.output)


if __name__ == "__main__":
    main()
