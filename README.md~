<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# CrobustaScreen
A pipeline for self-supervised phenotype detection from confocal images of *Ciona robusta* embryos.

![overview](https://github.com/ChristiaenLab/CrobustaScreen/blob/main/presentation/overview.png?raw=true)


# R Dependencies
`circlize`, `class`, `cluster`, `ComplexHeatmap`, `fgsea`, `ggplot2`, `ggpubr`, `igraph`, `keras`, `leiden`, `optparse`, `parallel`, `purrr`, `STRINGdb`, `umap` 

This pipeline also uses the custom packages [`dirfns`](https://github.com/kewiechecki/dirfns) and [`moreComplexHeatmap`](https://github.com/kewiechecki/moreComplexHeatmap)

# Usage
```
# parse segmentation data from /segdat
Rscript readEmbryos.R

# train autoencoder models
Rscript encode.R

# generate graph of known target protein interactions from STINGdb
Rscript interactions.R

# generate and score clusters for randomized hyperparameters
Rscript leiden.R

# generate plots and characterize optimal clusterings
Rscript plot.clust.R
```

# Preprocessing
The pipeline uses features extracted from segmentation of confocal images using Imaris. Summary statistics are extracted for segmened cells in each embryo. From the cell segmentation statistics 116 embryo-level parameters are computed. Parameters are normalized by z-score then scaled between -1 and 1.

![overview](https://github.com/ChristiaenLab/CrobustaScreen/blob/main/presentation/segmentation.png?raw=true)

# Dimension Reduction

![alt text](https://github.com/ChristiaenLab/CrobustaScreen/blob/main/fig/encode.dot.svg?raw=true)

Sample parameters are often strongly correlated. This is undesirable for self-supervised learning because each parameter additively contributes to distance used for clustering, resulting in disproportionate weight being given to phenotypes captured by multiple parameters. Linear methods of dimenison reduction (e.g. PCA) assume that all variables are independent and can be linearly combined. We could not assume that all of our measured input parameters were independent, so we instead used an autoencoder for dimension reduction.

An autoencoder is a neural network architecture widely used for denoising and image recognition. It works by encoding the input data into a lower dimensional representation that can be decoded with minimal loss. By extracting this lower dimensional encoding (the "bottleneck" or "embedding" layer), an autoencoder [can be used for dimension reduction](https://doi.org/10.1016/j.neucom.2015.08.104).
This results in an embedding that corresponds to the information content of the input data rather than absolute distance in phenotype space.

`encode.R` trains four autoencoders using embedding layers of 2, 3, 7, and 14 dimensions. `leiden.R` selects the optimal embedding based on [Akaike Information Criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion) defined as 

$$AIC = 2k - 2ln(\hat{L})$$ 

where $k$ is the number of parameters and $\hat{L}$ is a likelihood function, which we define as $1 - MSE$.

# Clustering Algorithm

![alt text](https://github.com/ChristiaenLab/CrobustaScreen/blob/main/fig/cluster.dot.svg?raw=true)

Euclidean distance between embeddings is used to compute a k-nearest neighbors graph. The graph is then partitioned into clusters by [modularity](https://en.wikipedia.org/wiki/Modularity_(networks)), which is defined as 

 $$\mathcal{H} = \frac{1}{2m}\sum_{c}( \,e_c - \gamma\frac{K_c^2}{2m}) $$

 where $m$ is the average degree of the graph, $e_c$ is the number of edges in cluster $c$, and $K_c$ is the number of nodes in cluster $c$. This equation has a nicely intuitive interpretation. Modularity $\mathcal{H}$ of a graph is given by the sum of how well-connected its clusters are, defined as the difference between the number of edges in the cluster and the expected number of edges given the number of nodes in the graph and average degree of a node.

Because optimizing modularity is NP-hard, we used the [leiden algorithm](https://arxiv.org/abs/1810.08473) to approximate an optimal solution.

Though modularity ensures that clusters are well-connected, the number of clusters returned is dependent on $\gamma$, which cannot be inferred from the data. A value of $k$ must also be selected for the input graph.

# Hyperparameter Selection

![alt text](https://github.com/ChristiaenLab/CrobustaScreen/blob/main/fig/sel.dot.svg?raw=true)

`leiden.R` performs clustering for 100 random $\gamma$ values between 0.01 and 3.0 for $k$ values ranging from 3 to 53.

We selected $k$ and $\gamma$ based on four validation metrics. Mean silhouette width was calculated from the euclidean distance etween embryos.

# $k$ Selection

**Ortholog Lookup**
`interactions.R` uses [STRINGdb](https://doi.org/10.1093/nar/gkq973) to construct a known protein interaction network of the perturbed genes. Because the *C. robusta* network is poorly characterized, we use ENSEMBL to obtain orthologs from *M. musculus* and *H. sapiens*. 

**GSEA**
The known protein interactions can be treated as a gene set for [GSEA](https://en.wikipedia.org/wiki/Gene_set_enrichment_analysis). Interactions can be ranked by edge count between embryos in two conditions. An enrichment score is calculated based on occurrence of known interactions near the top of the ranked list. An optimal $k$ can be selected by maximizing enrichment score.

**Gene Network**
A gene network can be created from the $k$-NN graph by drawing an edge between a pair of conditions if the $k$-NN graph is enriched in edges between embryos in that pair of conditions.
For each condition pair $(x,y)$, we use a hypergeometric test for enrichment of edges from embryos in $x$ to embryos in $y$.
We assume the null probability $p_{xy}$ to be given by

$$p_{xy}k = \frac{\binom{K_y}{k}\binom{M-K_y}{K_x-k}}{\binom{M}{K_x}}$$

where $K_y$ is the total degree of all nodes in $y$, $k$ is the number of edges from nodes in $x$ to nodes in $y$, $M$ is the total degree of all nodes in the graph, and $K_x$ is the total degree of all nodes in $x$. 
Effectively this means we consider all edges to be a population that the edges from nodes in $x$ are drawn from, and look for overrepresenation of edges connected to nodes in $y$. We consider a false disctovery rate of 0.05 to be significantly enriched. 
We define the odds ratio $OR_{xy}$ as

$$OR_{xy} = \frac{\frac{k}{K_x-k}}{\frac{K_y}{M-K_y}}$$

# $\gamma$ Selection
After selecting a $k$-NN graph, clustering is performed for randomized $\gamma$ values. Four metrics are calculated for each clustering: $log2(error)$, enrichment score, recall, and mean silhouette width. $\gamma$ is selected by optimizing for the product of these values.

**Reduced $k$-NN classifier**
A reduced $k$-NN classifier is created from a subset of the embeddings using the clusters as labels. The remaining embeddings are used as a test set. This process is repeated 1000 times per clustering to obtain a mean error.

**GSEA**
Condition pairs for each clustering are ranked by the proportion of edges that are between embryos in the same cluster vs. between embryos in different clusters. An enrichment score can be calculates as with $k$ selection.

**Comparison to Known Protein Interactions**
A second gene network is constructed using partial modularity between pairs of conditions. We define the partial modularity $H_{xy}$ of a condition pair $(x,y)$ as 

$$ H_{xy} = \,e_{xy} - \gamma\frac{K_x\,K_y}{2M} $$

where $e_{xy}$ is the total number of edges from embryos of condition $x$ to embryos of condition $y$, $K_x$ is the total degree of all embryos of condition $x$, $K_y$ is the total degree of all embryos in condition $y$, and M is the total degree of all nodes in the graph. If $H_{xy}$ is positive, we draw an edge between genes $x$ and $y$. We then calculate a recall score by comparing this graph to the graph of known protein interactions.

**Mean Silhouette Width**
Pointwise [silhouette width](https://doi.org/10.1016/0377-0427(87)90125-7) $s(i)$ is given by 

$$s(i) = \frac{b(i) - a(i)}{max[a(i),b(i)]}$$

where $a(i)$ is the average distance between node $i$ and other nodes in the same cluster, and $b(i)$ is the average distance between $i$ and other nodes in the closest other cluster.

# Cluster Characterization

`plot.clusts.R` tests for enrichment of experimental perturbations and experimenter-labeled phenotypes in each cluster using a hypergeometric test. For each condition $c$ in each cluster $x$, we assume the probability $p_{xc}$ of the intersect between $c$ and $x$ is given by

$$p_{xc}k = \frac{\binom{n_c}{k}\binom{N-n_c}{n_x-k}}{\binom{N}{n_x}}$$

where $k$ is the number of embryos in both $x$ and $c$, $n_c$ is the number of embryos in $c$, $N$ is the total number of embryos, and $n_x$ is the total number of embryos in $x$.

We define the odds ratio $OR_{xc}$ as

$$OR_{xc} = \frac{\frac{k}{n_x-k}}{\frac{n_c}{N-n_c}}$$
