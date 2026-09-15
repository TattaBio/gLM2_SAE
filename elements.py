"""Prodigal gene calling on DNA contigs, rendered as gLM2 tokens.

gLM2 reads a contig as an ordered mix of two modalities: coding (CDS, amino
acids, prefixed by a strand token) and intergenic (IGS, lowercase nucleotides,
always plus strand).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import pyrodigal

STRAND_TOKEN = {"+": "<+>", "-": "<->"}

# Per-token coordinate, for the intergenic tokens the SAE reads: (position,
# left_gene), where left_gene is the 0-based index of the nearest preceding CDS
# on the contig (-1 if none). None for every other token.
Coord = tuple[int, int] | None


@dataclass
class Element:
    kind: Literal["CDS", "IGS"]
    strand: Literal["+", "-"]
    start: int  # 1-based inclusive, plus strand
    end: int  # 1-based inclusive, plus strand
    body: str  # uppercase amino acids (CDS) or lowercase nucleotides (IGS)


@dataclass
class Contig:
    id: str
    elements: list[Element]

    @property
    def genes(self) -> list[Element]:
        return [e for e in self.elements if e.kind == "CDS"]


def read_fasta(path: str) -> Iterator[tuple[str, str]]:
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def gene_finder() -> pyrodigal.GeneFinder:
    """Prodigal in metagenomic mode, so contigs of any origin or length work."""
    return pyrodigal.GeneFinder(meta=True)


def build_contig(contig_id: str, seq: str, genes) -> Contig:
    """Interleave called genes with the intergenic stretches around them."""
    elements: list[Element] = []
    cursor = 1  # next unassigned 1-based position

    def add_igs(start: int, end: int) -> None:
        if end >= start:
            elements.append(Element("IGS", "+", start, end, seq[start - 1 : end].lower()))

    for gene in genes:
        add_igs(cursor, gene.begin - 1)
        # [begin, end] spans the stop codon, which is not part of the protein.
        protein = gene.translate(include_stop=False).upper()
        elements.append(Element("CDS", "+" if gene.strand > 0 else "-", gene.begin, gene.end, protein))
        cursor = max(cursor, gene.end + 1)
    add_igs(cursor, len(seq))
    return Contig(id=contig_id, elements=elements)


def tokenize_and_align(contig: Contig, tokenizer) -> tuple[list[int], list[Coord]]:
    """Token ids for the whole contig, plus a parallel per-token coordinate list.

    Each element becomes `[strand_token, *residue_tokens]`. Elements are
    tokenized separately (rather than as one concatenated string) so the
    alignment between tokens and coordinates cannot drift. Only intergenic
    tokens get a coordinate; coding tokens are context for the model, and the
    SAE never reads their positions.
    """
    strings = [STRAND_TOKEN[e.strand] + e.body for e in contig.elements]
    tokenized = tokenizer(strings)["input_ids"] if strings else []

    token_ids: list[int] = []
    coords: list[Coord] = []
    genes_seen = 0
    for element, toks in zip(contig.elements, tokenized, strict=True):
        assert len(toks) == 1 + len(element.body), "tokenizer emitted unexpected special tokens"
        left_gene = genes_seen - 1
        token_ids.append(toks[0])
        coords.append(None)
        for i, tok in enumerate(toks[1:]):
            token_ids.append(tok)
            coords.append((element.start + i, left_gene) if element.kind == "IGS" else None)
        if element.kind == "CDS":
            genes_seen += 1
    return token_ids, coords


def tile(token_ids: list[int], coords: list[Coord], context_size: int):
    """Non-overlapping windows of up to `context_size` tokens."""
    for start in range(0, len(token_ids), context_size):
        yield token_ids[start : start + context_size], coords[start : start + context_size]
