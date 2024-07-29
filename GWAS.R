library(rMVP)
args <- commandArgs(trailingOnly = T)
geno <- args[1]
pheno <- args[2]
if (grepl('vcf',geno,ignore.case=T)){
MVP.Data(fileVCF=geno,
         filePhe=pheno,
         fileKin=T,
         filePC=T,
         out=geno
         )
}else if (grepl('hmp',geno,ignore.case=T)){
MVP.Data(fileHMP=geno,
         filePhe=pheno,
         sep.hmp="\t",
         sep.phe="\t",
         SNP.effect="Add",
         fileKin=T,
         filePC=T,
         priority="memory",
         maxLine=10000,
         out=geno
         )
}else{return(message("Check input genotype fill!!!"))}

genotype <- attach.big.matrix(paste(geno,"geno.desc",sep='.'))
phenotype <- read.table(paste(geno,"phe",sep='.'),head=TRUE)
map <- read.table(paste(geno,"geno.map",sep='.') , head = TRUE)
Kinship <- attach.big.matrix(paste(geno,"kin.desc",sep='.'))
Covariates_PC <- bigmemory::as.matrix(attach.big.matrix(paste(geno,"pc.desc",sep='.')))

for(i in 2:ncol(phenotype)){
  imMVP <- MVP(
    phe=phenotype[, c(1, i)],
    geno=genotype,
    map=map,
    K=Kinship,
    #CV.GLM=Covariates,
    CV.MLM=Covariates_PC,
    CV.FarmCPU=Covariates_PC,
    # nPC.GLM=5,
    # nPC.MLM=3,
    # nPC.FarmCPU=3,
    priority="speed",
    ncpus=2,
    vc.method="BRENT",
    maxLoop=10,
    method.bin="EMMA",
    #permutation.threshold=TRUE,
    #permutation.rep=100,
    threshold=29.1,#0.05 #1(4.46)
    method=c( "FarmCPU","MLM"),
    file.output=T,
    file.type="pdf"
  )
  gc()
}

