#-------------------------------------------------------------------------------
# R scripts for demonstration of the algorithms taught in the course SIAT-BMI
# Teacher: Fengfeng Zhou
# Email: FengfengZhou@gmail.com
# Web: http://healthinformaticslab.org/course/SIAT-BMI/
# Semester: August 2014
# Version 1.0.1
# Update: 2014-08-24
# Dataset: GSE32175
#   Development of Transcriptomic Biomarker Signature in Human Saliva to Detect
#   Lung Cancer
#   URL: http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32175
# Restriction and disclaimer: 
#   These scripts are provided for the course project only. Please do not
#   re-distribute all these scripts, in part or completeness. For any other
#   purpose (including both commercial and non-profit purposes), please
#   contact the teacher for authorization and license.
#   
#   This section must be retained with all these course scripts.
#   
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#-------------------------------------------------------------------------------

# Initialization: loading the dataset
egMatrix <- read.csv("class-matrix-GSE32175.csv", header=TRUE, sep=",", row.names=1);
egClass <- read.csv("class-GSE32175-binary.csv", header=TRUE, sep=",", row.names=1);
indexP <- which(egClass$Class == "P");
indexN <- which(egClass$Class == "N");

egNumSmall <- nrow(egMatrix);

egSmallMatrix <- egMatrix[1:egNumSmall,];
dataRowNames <- row.names(egSmallMatrix);
resultMatrix <- matrix(nrow=nrow(egSmallMatrix),ncol=0);
#dataColNames <- c("#Feature");
dataColNames <- c();

# Two sample t-test
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class[c(indexP, indexN)]));

# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);

resultMatrix <- cbind(resultMatrix, dataFTest);
dataColNames <- cbind(dataColNames, "Ttest t", "Ttest Pvalue");

# Wilcoxon rank sum test
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) wilcox.test(x ~ egClass$Class[c(indexP, indexN)]));

# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);

resultMatrix <- cbind(resultMatrix, dataFTest);
dataColNames <- cbind(dataColNames, "Wtest W", "Wtest Pvalue");


# ANOVA test
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
### dataTest <- apply(egSmallMatrix, 1, function(x) anovalm((x[indexP], x[indexN])));
dataTest <- apply(egSmallMatrix, 1, function(x) anova(lm(x ~ egClass$Class[c(indexP, indexN)])));

# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(unlist(x[4]))[1], as.numeric(unlist(x[5]))[1]) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);

resultMatrix <- cbind(resultMatrix, dataFTest);
dataColNames <- cbind(dataColNames, "ANOVA F", "ANOVA Pr(>F)");

# Kruskal-Wallis rank sum test
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) kruskal.test(x ~ egClass$Class[c(indexP, indexN)]));

# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);

resultMatrix <- cbind(resultMatrix, dataFTest);
dataColNames <- cbind(dataColNames, "KS Chi", "KS Pvalue");

# Finalize the result matrix
#resultMatrix <- cbind( dataRowNames, resultMatrix);
#resultMatrix <- rbind( dataColNames, resultMatrix);
colnames(resultMatrix) <- dataColNames;
rownames(resultMatrix) <- dataRowNames;

# plot
boxplot(resultMatrix[,c(2, 4, 6, 8)]);

# integrated dot plots
par(mfrow=c(4,4));
indexCol <- 2*c(1:4);
for ( i in 1:4 )
  for (j in 1:4)
  {
    plot(resultMatrix[,indexCol[i]], resultMatrix[,indexCol[j]])
  }
par(mfrow=c(1,1));

# heatmap of the top 1000 rows, each statistical test for a column
# We cannot generate the heatmap for complete matrix, due to its memory requirement of 11.1Gb
library(gplots);
egRank <- rank(rank(resultMatrix[,2])+rank(resultMatrix[,4])+rank(resultMatrix[,6])+rank(resultMatrix[,8]));
indexTopRank <- which ( egRank <= 1000 ); # top 1000 features
heatmap.2(as.matrix(resultMatrix[indexTopRank,indexCol]), col=redgreen(75), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);

# heatmap of the top 1000 rows, each sample for a column
color.map <- function(tempClass)
{
  if( tempClass=='P' )
    'red'
  else
    'blue';
}
colorCol <- unlist(lapply(egClass$Class, color.map));
egRank <- rank(rank(resultMatrix[,2])+rank(resultMatrix[,4])+rank(resultMatrix[,6])+rank(resultMatrix[,8]));
indexTopRank <- which ( egRank <= 1000 ); # top 1000 features
heatmap.2(as.matrix(egSmallMatrix[indexTopRank,]), ColSideColors=colorCol, col=redgreen(75), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);

# heatmap of the top 50 rows, each sample for a column
color.map <- function(tempClass)
{
  if( tempClass=='P' )
    'red'
  else
    'blue';
}
colorCol <- unlist(lapply(egClass$Class, color.map));
egRank <- rank(rank(resultMatrix[,2])+rank(resultMatrix[,4])+rank(resultMatrix[,6])+rank(resultMatrix[,8]));
indexTopRank <- which ( egRank <= 50 ); # top 50 features
heatmap.2(as.matrix(egSmallMatrix[indexTopRank,]), ColSideColors=colorCol, col=redgreen(75), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);

# heatmap of the top 5 rows, each sample for a column
color.map <- function(tempClass)
{
  if( tempClass=='P' )
    'red'
  else
    'blue';
}
colorCol <- unlist(lapply(egClass$Class, color.map));
egRank <- rank(rank(resultMatrix[,2])+rank(resultMatrix[,4])+rank(resultMatrix[,6])+rank(resultMatrix[,8]));
indexTopRank <- which ( egRank <= 5 ); # top 50 features
resultMatrix[indexTopRank,];
heatmap.2(as.matrix(egSmallMatrix[indexTopRank,]), ColSideColors=colorCol, col=redgreen(75), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);

## histogram
par(mfrow=c(2,2));
hist(resultMatrix[,2], col="gray", labels=TRUE, border="black", breaks=10, main="Two sample t-test");
hist(resultMatrix[,4], col="gray", labels=TRUE, border="black", breaks=10, main="Wilcoxon rank sum test");
hist(resultMatrix[,6], col="gray", labels=TRUE, border="black", breaks=10, main="ANOVA test");
hist(resultMatrix[,8], col="gray", labels=TRUE, border="black", breaks=10, main="Kruskal-Wallis rank sum test");
par(mfrow=c(1,1));


