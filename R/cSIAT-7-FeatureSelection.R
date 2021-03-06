#-------------------------------------------------------------------------------
# R scripts for demonstration of the algorithms taught in the course SIAT-BMI
# Teacher: Fengfeng Zhou
# Email: FengfengZhou@gmail.com
# Web: http://healthinformaticslab.org/course/SIAT-BMI/
# Semester: August 2014
# Version 1.0.1
# Update: 2014-09-07
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

egNumSmall <- 1000;
#egNumSmall <- nrow(egMatrix);

# debugging code
egSmallMatrix <- egMatrix[1:egNumSmall,];
dataRowNames <- row.names(egSmallMatrix);
resultMatrix <- matrix(nrow=nrow(egSmallMatrix),ncol=0);
#dataColNames <- c("#Feature");
dataColNames <- c();

egRankingFeatureNumber <- 10;
egMaxFeatureNumber <- 100;
color.map <- function(tempClass)
{
    if( tempClass=='P' )
        'red'
    else
        'blue';
}
colorCol <- unlist(lapply(egClass$Class, color.map));


# Working
library(MASS)
library(glmnet);#lasso
library(kernlab);#svm
library(rpart);#dtree
library(e1071);#bayes
library(pamr);#pam
library(class); ##K-NN
library(minerva);#mine
library(FSelector);#best.first
library(RRF);
library(genefilter);
library(caret); # createFolds()

### Definitions of all the functions
#! k-Fold Cross Validation
efKFCV <- function(xx,yy,nfold,method)
{
    num_tp=num_fp=num_fn=num_tn=0
    index=NULL
    predy=NULL
    id=createFolds(yy, k = nfold, list = TRUE, returnTrain = T)
    rawdata=cbind(xx,yy)
    n=nrow(rawdata)
    p=ncol(rawdata)
    
    tPrediction <- rep(-1, nrow(rawdata));
    
    for (i in 1:nfold){
        # print(paste("Fold",i,sep=' '))
        index=id[[i]]
        y_train=rawdata[index,p]
        y_test=rawdata[-index,p]
        x_train=matrix(rawdata[index,-p],nrow=length(index))
        x_test=matrix(rawdata[-index,-p],nrow=(n-length(index)))
        
        predy <- efClassifier(x_train,y_train,x_test,method)
        
        num_tp[i]=sum(y_test==1 & predy==1)
        num_fn[i]=sum(y_test==1 & predy==0)
        num_fp[i]=sum(y_test==0 & predy==1)
        num_tn[i]=sum(y_test==0 & predy==0)
        
        tPrediction[-index] <- predy;
    }
    se=sum(num_tp)/sum(yy==1)
    sp=sum(num_tn)/sum(yy==0)
    acc=sum(num_tp+num_tn)/length(yy)
    avc=(se+sp)/2
    mcc=(sum(num_tp)*sum(num_tn)-sum(num_fp)* sum(num_fn))/
        (sqrt((sum(num_tp)+sum(num_fp))*(sum(num_tp)+sum(num_fn))*(sum(num_tn)+sum(num_fp))*(sum(num_tn)+sum(num_fn))))
    out=round(cbind(se,sp,acc,avc,mcc),3)
    return(list(out=out, prediction=tPrediction))
}
#! Standard IO of all the investigated classifiers
efClassifier <- function(x_train,y_train,x_test,method)
{
    if (method=="SVM"){
        fit=ksvm(x_train,y_train,type="C-svc",kernel="rbfdot")
        predy=predict(fit,x_test)
    } else { 
        if (method=="NBayes"){
            colnames(x_train)=NULL
            colnames(x_test)=NULL
            data_train=data.frame(ex=x_train,ey=as.factor(y_train))
            data_test=with(data_train,data.frame(ex=x_test))
            fit <- naiveBayes(ey~.,data=data_train)
            predy=predict(fit, data_test, type="class")
        } else {
            if (method=="DTree"){
                colnames(x_train)=NULL
                colnames(x_test)=NULL
                data_train=data.frame(ex=x_train,ey=as.factor(y_train))
                data_test=with(data_train,data.frame(ex=x_test))
                fit <- rpart(ey~.,data=data_train)
                predy=predict(fit, data_test, type="class")
            } else {
                if (method=="Lasso"){
                    cv.fit <- cv.glmnet(x_train, y_train, family = "binomial")
                    fit <- glmnet(x_train, y_train, family = "binomial")
                    pfit = predict(fit,x_test,s = cv.fit$lambda.min,type="response")
                    predy<-ifelse(pfit>0.5,1,0)
                } else { 
                    if (method=="KNN"){
                        predy<-knn1(x_train,x_test,y_train)
                    }
                }    
            }
        }
    }
    return (predy)
}

efClassToInt<-function(classes)
{
    levelsClass <- sort(levels(as.factor(classes)));
    #for(i in levelsClass)
    for(i in levelsClass)
    {
        classes<-replace(classes,classes==i,match(i,levelsClass)-1);
        #classes <- replace(classes, classes==levelsClass[i], i-1);
    }
    classes <- (as.numeric(classes));
    return (classes);
}

efBinaryPerformance <- function(tClass, tPrediction)
{
    tTP <- sum(tClass==1 & tPrediction==1 );
    tFN <- sum(tClass==1 & tPrediction==0 );
    tFP <- sum(tClass==0 & tPrediction==1 );
    tTN <- sum(tClass==0 & tPrediction==0 );
    tSn <- tTP/(tTP+tFN);
    tSp <- tTN/(tTN+tFP);
    tAcc <- (tTP+tTN)/(tTP+tFN+tTN+tFP);
    tAvc <- (tSn+tSp)/2;
    tMCC <- (tTP*tTN-tFP*tFN)/sqrt((tTP+tFP)*(tTP+tFN)*(tTN+tFP)*(tTN+tFN));
    return(round(cbind(tSn, tSp, tAcc, tAvc, tMCC), 3));
}

efPerformanceMatrix <- function( kMatrix )
{
    tresultMatrix <- matrix(nrow=0,ncol=5);
    tdataRowNames <- c();
    tdataColNames <- c("Sn", "Sp", "Acc", "Avc", "MCC");
    
    # SVM
    egResult <- efKFCV(t(kMatrix), egClassLabel, 3, "SVM");
    etMeasurements <- egResult$out;
    tresultMatrix <- rbind(tresultMatrix, etMeasurements[1,]);
    tdataRowNames <- c(tdataRowNames, "SVM");
    
    # NBayes
    egResult <- efKFCV(t(kMatrix), egClassLabel, 3, "NBayes");
    etMeasurements <- egResult$out;
    tresultMatrix <- rbind(tresultMatrix, etMeasurements[1,]);
    tdataRowNames <- c(tdataRowNames, "NBayes");
    
    # DTree
    egResult <- efKFCV(t(kMatrix), egClassLabel, 3, "DTree");
    etMeasurements <- egResult$out;
    tresultMatrix <- rbind(tresultMatrix, etMeasurements[1,]);
    tdataRowNames <- c(tdataRowNames, "DTree");
    
    # Lasso
    egResult <- efKFCV(t(kMatrix), egClassLabel, 3, "Lasso");
    etMeasurements <- egResult$out;
    tresultMatrix <- rbind(tresultMatrix, etMeasurements[1,]);
    tdataRowNames <- c(tdataRowNames, "Lasso");
    
    # KNN
    egResult <- efKFCV(t(kMatrix), egClassLabel, 3, "KNN");
    etMeasurements <- egResult$out;
    tresultMatrix <- rbind(tresultMatrix, etMeasurements[1,]);
    tdataRowNames <- c(tdataRowNames, "KNN");
    
    rownames(tresultMatrix) <- tdataRowNames;
    colnames(tresultMatrix) <- tdataColNames;

    return ( tresultMatrix );
}

### End of function definition


#memory.limit(4000)

# Individual Ranking: Two sample t-test
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class[c(indexP, indexN)]));
# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);
library(gplots);
egRank <- rank(dataFTest[,2]);
indexTopRank <- which ( egRank <= egRankingFeatureNumber ); # top 10 features
heatmap.2(as.matrix(egSmallMatrix[indexTopRank,]), ColSideColors=colorCol, col=redgreen(75), main="T-test top 10 features", margins=c(10,7), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);
set.seed(0);
egClassLabel <- efClassToInt(as.numeric(egClass$Class));
resultMatrix <- efPerformanceMatrix(egSmallMatrix[indexTopRank,]);
par(mar=c(3, 2, 1.2, 2)); # bottom, left, top, right
barplot(resultMatrix, beside=TRUE, col=rainbow(5), main="T-test top 10 features", ylim=c(-0.2, 1.0), xlim=c(0, 36));
#legend(30, 1.0, rownames(egSmallMatrix[indexTopRank,]), cex=1.0, bty="n", fill=rainbow(5));
legend(30, 1.0, rownames(resultMatrix), cex=1.0, bty="n", fill=rainbow(5));
box();


# Individual Ranking: Wilcoxon test
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) wilcox.test(x ~ egClass$Class[c(indexP, indexN)]));
# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);
library(gplots);
egRank <- rank(dataFTest[,2]);
indexTopRank <- which ( egRank <= egRankingFeatureNumber ); # top 10 features
heatmap.2(as.matrix(egSmallMatrix[indexTopRank,]), ColSideColors=colorCol, col=redgreen(75), main="Wilcoxon-test top 10 features", margins=c(10,7), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);
set.seed(0);
egClassLabel <- efClassToInt(as.numeric(egClass$Class));
resultMatrix <- efPerformanceMatrix(egSmallMatrix[indexTopRank,]);
par(mar=c(3, 2, 1.2, 2)); # bottom, left, top, right
barplot(resultMatrix, beside=TRUE, col=rainbow(5), main="Wilcoxon-test top 10 features", ylim=c(-0.2, 1.0), xlim=c(0, 36));
#legend(30, 1.0, rownames(egSmallMatrix[indexTopRank,]), cex=1.0, bty="n", fill=rainbow(5));
legend(30, 1.0, rownames(resultMatrix), cex=1.0, bty="n", fill=rainbow(5));
box();



# Group optimization: PAM
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) wilcox.test(x ~ egClass$Class[c(indexP, indexN)]));
# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);
library(gplots);
egRank <- rank(dataFTest[,2]);
indexTopRank <- which ( egRank <= egRankingFeatureNumber ); # top 10 features
dataRowNames <- rownames(egSmallMatrix);
dataColNames <- colnames(egSmallMatrix);

set.seed(0);
PAMdata <- list(x=as.matrix(egSmallMatrix), genenames=paste("g",dataRowNames), geneid=paste("g",dataRowNames), y=factor(egClass$Class));
PAMtrain<- pamr.train(PAMdata);
PAMcv <- pamr.cv(PAMtrain,PAMdata);

PAMpdex1<-max(which(PAMcv$error==min(PAMcv$error)));
PAMpdex2<-max(which(PAMcv$size!=0));
PAMpdex<-ifelse (PAMcv$size[PAMpdex1]==0,PAMpdex2,PAMpdex1);
PAMcvthreshold<-PAMcv$threshold[PAMpdex];

PAMglist <- pamr.listgenes(PAMtrain, PAMdata, PAMcvthreshold, fitcv=PAMcv);
egChosenFeatures <- sub("^g ", "", PAMglist[,1]);
if( length(egChosenFeatures)>egMaxFeatureNumber )
{
    egChosenFeatures <- egChosenFeatures[1:egMaxFeatureNumber];
}


heatmap.2(as.matrix(egSmallMatrix[egChosenFeatures,]), ColSideColors=colorCol, col=redgreen(75), main=paste("PAM chosen ", length(egChosenFeatures), " features"), margins=c(10,7), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);
set.seed(0);
egClassLabel <- efClassToInt(as.numeric(egClass$Class));
resultMatrix <- efPerformanceMatrix(egSmallMatrix[egChosenFeatures,]);
par(mar=c(3, 2, 1.2, 2)); # bottom, left, top, right
barplot(resultMatrix, beside=TRUE, col=rainbow(5), main=paste("PAM chosen ", length(egChosenFeatures), " features"), ylim=c(-0.2, 1.0), xlim=c(0, 36));
legend(30, 1.0, rownames(resultMatrix), cex=1.0, bty="n", fill=rainbow(5));
box();



# Group optimization: PAM
#! N vs P
# dataTest <- apply(egSmallMatrix, 1, function(x) t.test(x ~ egClass$Class));
dataTest <- apply(egSmallMatrix, 1, function(x) wilcox.test(x ~ egClass$Class[c(indexP, indexN)]));
# retrieved values: t and P-value
dataFTest <- lapply( dataTest, function(x) c(as.numeric(x[1]), as.numeric(x[3])) );
dataFTest <- unlist(dataFTest);
dim(dataFTest) <- c(2, egNumSmall);
dataFTest <- t(dataFTest);
library(gplots);
egRank <- rank(dataFTest[,2]);
indexTopRank <- which ( egRank <= egRankingFeatureNumber ); # top 10 features
dataRowNames <- rownames(egSmallMatrix);
dataColNames <- colnames(egSmallMatrix);

set.seed(0);
RRFrrf <- RRF(as.matrix(t(egSmallMatrix)),y=as.factor(egClass$Class));
RRFimp<-RRFrrf$importance;
RRFimp<-RRFimp[,"MeanDecreaseGini"];
fRRF <- which(RRFimp>0);  ##FS index
egChosenFeatures <- dataRowNames[fRRF];
if( length(egChosenFeatures)>egMaxFeatureNumber )
{
    egChosenFeatures <- egChosenFeatures[1:egMaxFeatureNumber];
}

heatmap.2(as.matrix(egSmallMatrix[egChosenFeatures,]), ColSideColors=colorCol, col=redgreen(75), main=paste("RRF chosen ", length(egChosenFeatures), " features"), margins=c(10,7), scale="none", key=TRUE, symkey=FALSE, density.info="none", trace="none", cexRow=1, cexCol=1);
set.seed(0);
egClassLabel <- efClassToInt(as.numeric(egClass$Class));
resultMatrix <- efPerformanceMatrix(egSmallMatrix[egChosenFeatures,]);
par(mar=c(3, 2, 1.2, 2)); # bottom, left, top, right
barplot(resultMatrix, beside=TRUE, col=rainbow(5), main=paste("RRF chosen ", length(egChosenFeatures), " features"), ylim=c(-0.2, 1.0), xlim=c(0, 36));
legend(30, 1.0, rownames(resultMatrix), cex=1.0, bty="n", fill=rainbow(5));
box();
