source("R/clustplots.R")

library(purrr)
library(optparse)
library(umap)
library(ggplot2)
library(moreComplexHeatmap)

parser <- OptionParser()
parser <- add_option(parser, '--dir', action = 'store',
		     default='data')
parser <- add_option(parser, '--embeddingdir', action = 'store',
		     default = 'data/embedding')
parser <- add_option(parser, '--clustdir', action = 'store',
		     default = 'data/clust')
parser <- add_option(parser, '--clusts', action = 'store',
		     default = 'data/clusts.csv')
parser <- add_option(parser, '--groups', action = 'store',
		     default = 'data/groups.csv')
parser <- add_option(parser, '--pheno', action = 'store',
		     default = 'data/phenotype.csv')
opts <- parse_args(parser)


groups <- read.csv(opts$groups)
pheno <- read.csv(opts$pheno,row.names=1)
pheno <- cbind(groups[,"Condition",drop=F],pheno)

clusts <- read.csv(opts$clusts)
names(clusts) <- sub("Column","Head",names(clusts))

E <- lrtab(opts$embeddingdir,read.csv)
C <- lrtab(opts$clustdir,read.csv)

E_c  <- read.csv(paste0(opts$dir,'/E_consensus.csv'))
C_c  <- read.csv(paste0(opts$dir,'/C_consensus.csv'))
logits_c <- log(C_c/(1-C_c))
clusts_c  <- read.csv(paste0(opts$dir,'/clusts_consensus.csv'))

umap_E  <-  as.data.frame(umap(E_c)$layout)
names(umap_E) <- c("UMAP1","UMAP2")
umap_E <- cbind(umap_E,Cluster=as.character(clusts_c[,1]))
umap_E <- cbind(umap_E,groups)

plot_E  <- lapply(c("Cluster","Phenotype","Condition"),
		  function(col){
			  g <- ggplot(umap_E,aes_string(x="UMAP1",y="UMAP2",col=col)) + geom_point()
			  return(g)
		  })
arrange.plots(plot_E,c("","",""),out="data/consensus_umap")

plots <- lapply(names(E),function(i){
    dat <- as.data.frame(umap(E[[i]])$layout)
    names(dat) <- c("UMAP1","UMAP2")
    dat <- cbind(dat,col=groups$Phenotype)
    g <- ggplot(dat,aes_string(x="UMAP1",y="UMAP2",col="col")) + geom_point()
    return(g)
    })

arrange.plots(plots,names(clusts),out="data/E_umap")

plots <- lapply(names(E),function(i){
    dat <- as.data.frame(prcomp(E[[i]])$x)
    names(dat) <- c("PC1","PC2")
    dat <- cbind(dat,col=groups$Phenotype)
    g <- ggplot(dat,aes_string(x="PC1",y="PC2",col="col")) + geom_point()
    return(g)
    })

arrange.plots(plots,names(clusts),out="data/E_pca")

csplit <- do.call(c,lapply(names(clusts),rep,14))

embeddings <- do.call(cbind,E)

pdf("data/embeddings.pdf")
Heatmap(embeddings,
        name="embedding",
        show_column_names=F,
        row_split=groups$Phenotype,
        column_split=csplit,
        row_title_rot=0,
        column_title_rot=90)
dev.off()

pdf("data/embedding_consensus.pdf")
Heatmap(E_c,
        name="embedding",
        show_column_names=F,
        row_split=groups$Phenotype,
        row_title_rot=0,
        column_title_rot=90)
dev.off()

softclust <- do.call(cbind,C)
logits <- log(softclust/(1-softclust))
csplit <- do.call(c,lapply(names(clusts),rep,14))

pdf("data/logits.pdf")
Heatmap(logits,
        name="logit",
        show_column_names=F,
        row_split=groups$Phenotype,
        column_split=csplit,
        row_title_rot=0,
        column_title_rot=90)
dev.off()

pdf("data/logits_consensus.pdf")
Heatmap(logits_c,
        name="logit",
        show_column_names=F,
        row_split=groups$Phenotype,
        row_title_rot=0,
        column_title_rot=90)
dev.off()


sapply(names(C), function(i){
    pdf(paste0('data/clust/',i,'.pdf'))
    draw(Heatmap(C[[i]]))
    dev.off()
    })

hyper <- lapply(clusts,
                function(c) lapply(as.data.frame(pheno),
                                   function(p) condHyper(row.names(pheno),
                                                         p,c)))

hyperattr <- function(hyper,attr) {
    lapply(hyper,function(h) do.call(rbind, lapply(h,'[[', attr)))
    }

odds <- hyperattr(hyper,"log2OR")

csplit <- do.call(c,mapply(rep,names(odds),sapply(odds,dim)[2,]))

odds <- do.call(cbind,odds)
fdr <- do.call(cbind,hyperattr(hyper,'FDR'))
qval <- do.call(cbind,hyperattr(hyper,'q'))

# split phenotype & condition into separate panels
rowsplit <- unlist(mapply(function(x,y) rep(y,nrow(x$log2OR)),
                hyper[[1]],
                names(hyper[[1]])))

dotPscale(as.matrix(odds), 
        as.matrix(fdr), 
        as.matrix(qval), 
        file='phenotype', 
        path='data', 
        row_split=rowsplit, 
    column_split=csplit,
    row_title_rot=0,
    column_title_rot=90)

clusthyper <- function(dat, clusts, out){
	require(moreComplexHeatmap)
	# run hypergeometric tests for enrichment of conditions and phenotypes
	hyper <- lapply(dat, 
			function(x) condHyper(row.names(dat),
					      x,clusts))

	# extract fields from test & reformat as matrices
	odds <- do.call(rbind,lapply(hyper,'[[','log2OR'))
	fdr <- do.call(rbind,lapply(hyper,'[[','FDR'))
	qval <- as.matrix(do.call(rbind,
				  lapply(hyper,'[[','q')))

	# split phenotype & condition into separate panels
	#         if(length(dat)>1){
	rowsplit <- unlist(
		mapply(
		       function(x,y) rep(y,nrow(x$log2OR)),
		       hyper,
		       names(hyper)
		)
	)
	#         } else rowsplit <- NULL

	# write matrices to csv
	dir.csv(cbind(
		parameter=rowsplit,
		condition=row.names(odds),
		as.data.frame(odds)
	),'log2OR', out, append.date=F)
	dir.csv(cbind(
		parameter=rowsplit,
		condition=row.names(odds),
		as.data.frame(fdr)
	),'FDR', out, append.date=F)
	dir.csv(cbind(
		parameter=rowsplit,
		condition=row.names(odds),
		as.data.frame(qval)
	),'size', out, append.date=F)

	if(length(unique(clusts))>1){
		dotPscale(
			odds, 
			fdr, 
			qval, 
			file='condition', 
			path=out, 
			row_split=rowsplit, 
			row_title_rot=0
		)
	}
}

hyper_c <- lapply(as.data.frame(pheno), 
		  function(p) condHyper(row.names(pheno), 
					p,clusts_c))



odds_c <- do.call(rbind,sapply(hyper_c,'[',"log2OR"))
fdr_c <- do.call(rbind,sapply(hyper_c,'[','FDR'))
qval_c <- do.call(rbind,sapply(hyper_c,'[','q'))

# split phenotype & condition into separate panels
rowsplit_c <- unlist(mapply(function(x,y) rep(y,nrow(x$log2OR)),
                hyper_c,
                names(hyper_c)))

dotPscale(as.matrix(odds_c), 
        as.matrix(fdr_c), 
        as.matrix(qval_c), 
        file='phenotype_c', 
        path='data', 
        row_split=rowsplit_c, 
    row_title_rot=0,
    column_title_rot=90)

cond_c <- condHyper(row.names(pheno),groups$Condition,clusts_c)

dotPscale(as.matrix(cond_c$log2OR), 
        as.matrix(cond_c$FDR), 
        as.matrix(cond_c$q), 
        file='cond_c', 
        path='data', 
    row_title_rot=0,
    column_title_rot=90)

