<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# MSDeePWAK
A pipeline for self-supervised phenotype detection from confocal images of *Ciona robusta* embryos.
It uses ensemble [DeePWAK](https://github.com/kewiechecki/DeePWAK) clustering to find the optimal way of dividing the data into phenotypes.

![overview](https://github.com/kewiechecki/MSDeePWAK/blob/main/presentation/overview.png?raw=true)


# R Dependencies
`circlize`, `class`, `cluster`, `ComplexHeatmap`, `ggplot2`, `ggpubr`, `igraph`, `leiden`, `optparse`, `parallel`, `purrr`, `umap` 

This pipeline also uses the custom packages [`dirfns`](https://github.com/kewiechecki/dirfns) and [`moreComplexHeatmap`](https://github.com/kewiechecki/moreComplexHeatmap)

# Usage
The pipeline expects features extracted from Imaris to be found in `segdat/`. 
```
# preprocess segmentation data in segdat/
Rscript preprocess.R
```
The clustering script reads data from `data/z_dat.csv`.
Hyperparameters in `deepwak.jl` are tuneable. See below for a description.
```
# ensemble DeePWAK clustering
julia deepwak.jl
```
The pipeline expects a table `data/groups.csv` with `Condition` and `Phenotype` columns.
It also expects a table `data/phenotype.csv` for generating phenotype enrichment plots.
The columns should correspond to phenotypic categories and the values of a column should be the possible phenotypes of that category.
```
# enrichment tests and figures
Rscript enrichment.R
```

# Preprocessing
The pipeline uses features extracted from segmentation of confocal images using Imaris. Summary statistics are extracted for segmened cells in each embryo. From the cell segmentation statistics 116 embryo-level parameters are computed. Parameters are normalized by z-score then scaled between -1 and 1.

![overview](https://github.com/ChristiaenLab/CrobustaScreen/blob/main/presentation/segmentation.png?raw=true)

#DeePWAK Overview

[DeePWAK](https://github.com/kewiechecki/DeePWAK/blob/master/paper/DeePWAK.pdf) treats clustering as a denoising optimization problem. Taking inspiration from [DEWAKSS](https://nyuscholars.nyu.edu/en/publications/optimal-tuning-of-weighted-knn-and-diffusion-based-methods-for-de), we use graph diffusion to find an optimal representation.

