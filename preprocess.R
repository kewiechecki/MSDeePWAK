source("R/imageFns.R")
library(dirfns)

dirs <- list.files(
  "segdat",
  pattern = "_Statistics$",
  recursive = T,
  include.dirs = T,
  full.names = T
)

#sgrna <- read.csv("sgRNA.csv")

meta <- strsplit(dirs,'/')

meta <- data.frame(
  condition=sub('_','.',sub(' .*','',sapply(meta,'[',3))),
  grouping=dirs,
  batch=sapply(meta,'[',2),
  slide='',
  date=sub('.*([0-9]{2})([0-9]{2})([0-9]{4})','\\3_\\1_\\2',sapply(meta,'[',2)),
  #date=sub('.*([0-9]{2})([0-9]{2})([0-9]{4})_\\[.*','\\3_\\1_\\2',sapply(meta,'[',4)),
  image=paste0('image',sub('.*Image_([0-9]+)\\].*','\\1',dirs)),
  half='',
  file=dirs,
  stringsAsFactors = F
)
sel <- grep('slide',dirs)
meta$slide[sel] <- sub('.* (slide)\\s?([0-9]+).*','\\1\\2',dirs[sel])

meta[grep('next part',dirs),'slide'] <- 'nextpart'
meta[grep('right',dirs,ignore.case=T),'half'] <- "right"
meta[grep('left',dirs,ignore.case=T),'half'] <- "left"

sel <- grep('normal',meta$batch)
meta[sel,'batch'] <- 'old'
meta[-sel,'batch'] <- 'new'

meta$cond <- sub('^[0-9]+[A-Z]\\-','',meta$condition)
meta[meta$cond=="EPH",'cond'] <- "Eph"
meta[meta$cond=="GNA12.13",'cond'] <- "Gna12.13"
meta[meta$cond=="GNA.L.S",'cond'] <- "Gna.L.S"

#tmp <- sgrna[!duplicated(sgrna[,c(1,3)]),c(1,3)]
#sapply(unique(meta$cond),grep,sgrna$Gene.Name,ignore.case=T)

meta$id <- sub('^_','',sub('_$','',paste(meta$slide,meta$image,meta$half,sep='_')))

meta[grep('grouped',meta[,2],ignore.case = T),2] <- 'surface'
meta[grep('surf[az]ce',meta[,2],ignore.case = T),2] <- 'surface'
meta[grep('individual',meta[,2],ignore.case = T),2] <- 'cell'
meta[grep('nuclei',meta[,2],ignore.case = T),2] <- 'cell'
meta[grep('cell',meta[,2],ignore.case = T),2] <- 'cell'


meta$name <- make.names(paste(
			      meta$cond,meta$id,
			      meta$date,
			      sep='.'))

embryos <- split(meta[,c('name','file')],meta$grouping)
embryos <- lapply(embryos,function(x) data.frame(name=make.unique(x$name),file=x$file))

#hack to fix names
sel <- setdiff(embryos$cell$name,embryos$surface$name)
embryos$cell$name[embryos$cell$name%in%sel] <- setdiff(embryos$surface$name,embryos$cell$name)

embryos <- do.call(merge,append(unname(embryos),list(by='name')))
names(embryos) <- c('name','cell','surface')

celldat <- lapply(embryos$cell,readEmbryos)
surfacedat <- lapply(embryos$surface,readEmbryos)


celldat <- lapply(celldat,normvol)
ndists <- sapply(celldat,getDists)
cdists <- sapply(celldat,getDists,F)
stats <- sapply(celldat,groupStats)
dists <- rbind(ndists,cdists,stats)

cellID <- sapply(celldat,function(x) factor(x[,2],levels=c("ATM","TVC")))
ncell <- sapply(cellID,table)
row.names(ncell) <- c('nATM','nTVC')
cell <- rbind(dists,ncell)
colnames(cell) <- embryos$name

membrane <- sapply(surfacedat,membraneStats)
colnames(membrane) <- embryos$name

params <- merge(t(cell),t(membrane),0)
params[sapply(params,function(x) !is.finite(x)&is.numeric(x))] <- 0
dir.csv(params, 'params', 'out', row.names=F, append.date=F)

z <- zdf(params[,-1])
z <- cbind(Row.names=params[,1],as.data.frame(z))
dir.csv(z, 'z_dat', 'out', append.date=F, row.names=F)

