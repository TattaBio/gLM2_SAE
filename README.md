# gLM2 SAE

Inference code for identifying intergenic sequence features using a sparse autoencoder
trained on gLM2-650M representations.

Model weights: [tattabio/gLM2_650M_sae](https://huggingface.co/tattabio/gLM2_650M_sae).

## Usage

Predict intergenic SAE features from a DNA fasta:

```bash
uv run predict.py contigs.fna -o features.csv -g genes.csv
```

From a DNA FASTA this calls genes with prodigal, encodes each contig as a gLM2 input string (coding regions as amino acids, intergenic regions as nucleotides, in contig order),
runs it through the model, and encodes the layer-24 residuals at the intergenic positions
with the SAE. Coding regions are still part of the forward pass but are just not encoded by the SAE, which was trained
on intergenic positions only.

`features.csv` has one row per feature run:

| column | meaning |
| --- | --- |
| `contig_id` | FASTA record name |
| `feature_id` | SAE latent, 0–16383 |
| `start`, `end`, `length` | 1-based inclusive contig coordinates of the run |
| `n_active` | positions inside the run where the latent fired |
| `max_activation`, `total_activation` | peak and summed activation over those positions |
| `left_gene`, `right_gene` | 1-based indices of the flanking genes (blank at a contig end), matching `gene_index` in the `-g` output |

A single firing position is noise, so a latent is only reported where it fires across a
run: at least `--min-run` (8) positions spanning at least that many bp, tolerating gaps of
up to `--run-gap` (3) bp. Latents firing on more than `--max-density` (0.5) of a window's
intergenic positions are dropped as ubiquitous.
