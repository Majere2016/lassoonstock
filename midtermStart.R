# Data analysis

library(readtext)
library(SnowballC)
library(tidytext)

setwd("C:/Users/WOD/Desktop/lasso/lasso/")

# Let's get to know the data a bit

data<-readtext("RedditNews.csv",skip=1)

date<-data[2] # this is the day of the news

subset<-date=="7/1/16" # let's take a look at news headlines on 7/1/16

data[subset,3] # we have 23 news headlines


# Read the DJIA data

dj<-read.csv("DJIA.csv")

head(dj) # Open price, highest, lowest and close price

ndays<-nrow(dj) # 1989 days



# Read the words

words<-read.csv("WordsFinal.csv",header=F)

words<-words[,1]

head(words)


# Read the word-day pairings

doc_word<-read.table("WordFreqFinal.csv",header=F)

# Create a sparse matrix
# Fit a LASSO with all  product categories and  words

library(gamlr)
library(Matrix)
spm<-sparseMatrix(
		i=doc_word[,1],
		j=doc_word[,2],
		x=doc_word[,3],
		dimnames=list(id=1:ndays,words=words))

dim(spm)

# We select only words at occur at least 5 times

cols<-apply(spm,2,sum)

index<-apply(spm,2,sum)>5

spm<-spm[,index]

# and words that do not occur every day

index<-apply(spm,2,sum)<ndays
index_2<-apply(spm,2,sum)<ndays
spm<-spm[,index]
spm_3<-spm[,index_2]
dim(spm) # we end up with 3183 words


#  *** FDR *** analysis

spm<-spm[-ndays,]

time<-dj[-ndays,1]
# *** setting new fillter ***
spm_test <- spm[ndays,]
time_test <- dj[ndays,]
# Take returns 

par(mfrow=c(1,2))

R<-(dj[-ndays,7]-dj[-1,7])/dj[-1,7]

plot(R~time,type="l")



# Take the log of the maximal spread

V<-log(dj[-ndays,3]-dj[-ndays,4])

plot(V~time,type="l")


# FDR: we want to pick a few words that correlate with the outcomes (returns and volatility)

# create a dense matrix of word presence

P <- as.data.frame(as.matrix(spm>0))


# we will practice parallel computing now

library(parallel)


margreg <- function(x){
	fit <- lm(Outcome~x)
	sf <- summary(fit)
	return(sf$coef[2,4]) 
}

cl <- makeCluster(detectCores())

# pull out stars and export to cores


# **** Analysis for Returns ****

Outcome<-R

clusterExport(cl,"Outcome") 

# run the regressions in parallel

mrgpvals <- unlist(parLapply(cl,P,margreg))

# continue on your own

#1. watch the p-value of R and V
par(mfrow=c(1,2))

Rp <- t.test(R)
Vp <- t.test(V)

hist(R,freq = F,col = "red")
lines(density(R), col="blue", lwd=2)
abline(v=Rp$p.value,col = "blue", lwd = 3, lty = 2)
hist(V,freq = F,col="green")
lines(density(V), col="black", lwd=2)
abline(v=Vp$p.value,col = "black", lwd = 3, lty = 2)

#2.What is the alpha value (p-value cutoff) associated with 10% False Discovery Rate? 
#How many words are significant at this level? 
#(Again, analyze both outcomes V and R.) What are the advantages and disadvantages of FDR for word selection? (11 points)
summary(mrgpvals)
par(mfrow=c(1,2))
t_mr<-t.test(mrgpvals)
mrg_names <- data.frame(mrgpvals)
key_word<-mrg_names[which(mrg_names$mrgpvals<0.1),]
length(key_word)
hist(key_word)
hist(Outcome)
#好处在于过滤了很多没有意义和低效率的偏差字符串，坏处在于同时过滤了很多词并且破坏了原来的分布。
# **** Repeat for volatility

Outcome<-V

clusterExport(cl,"Outcome") 

# run the regressions in parallel

mrgpvals <- unlist(parLapply(cl,P,margreg))

# continue on your own
#3. Now, focus only on volatility V . Suppose you just mark the 20 smallest p-values as significant. 
#How many of these discoveries do you expect to be false? 
#Are the p-values independent? Discuss
mrg_V <- data.frame(mrgpvals)
mrg_V$rk<-rank(mrg_V$mrgpvals)
mrg_V[which(mrg_V$rk<=20),]
hist(mrg_V$mrgpvals)
#他们都相互独立，并且p值略大，所代表的词没有太大的相关性
# ***** LASSO analysis *****
par(mfrow=c(1,2))
# First analyze returns 

lasso1<- gamlr(spm, y=R, lambda.min.ratio=1e-3)
lasso1<- gamlr(spm, y=R, standardize=FALSE,family="binomial",
lambda.min.ratio=1e-3) # Let's fit the LASSO with just the product categories

# continue on your own
summary(lasso1)
hist(lasso1$lambda)
plot(lasso1)
dev <- lasso1$deviance[which.min(AICc(lasso1))]  # this is the deviance
#of the AICc selected model
dev0<- lasso1$deviance[1] # this is the null deviance
#(you could have fitted a null model and get it from glm, that is fine)
1-dev/dev0  # not much signal
# **** LASSO Analysis of volatility **** #

lasso2<- gamlr(spm, y=V, lambda.min.ratio=1e-3)

library(knitr)
# continue on your own
hist(lasso2$lambda)
plot(lasso2)
Betas <- drop(coef(lasso2))  # AICc default selection
length(Betas) # intercept product cats words
len_b <- length(Betas)
sum(Betas[251:len_b]!=0)  # predictive words
#choose 10 most positive review words in this model
o <- order(Betas[251:len_b],decreasing = T)
kable(Betas[251:len_b][o[1:10]])
# What is the in-sample R2 now
1- lasso2$deviance[which.min(AICc(lasso2))]/lasso2$deviance[1]

exp(Betas[names(Betas)=="terrorist"])
# each extra word discount increases the odds of 5 stars 1000 times
# the probability of 5 star when the review consists of just one word "terrorist" is
linpred<-as.numeric(Betas[1]+Betas[names(Betas)=="terrorist"])
kable(exp(linpred)/(1+exp(linpred))) # nearly 1
dev.off()

# let's try to predict future volatility from past volatility, we will add one more predictor-> volatility from the previous days


Previous<-log(dj[-1,3]-dj[-1,4]) # remove the last return

spm2<-cbind(Previous,spm) # add the previous return to the model matrix

colnames(spm2)[1]<-"previous" # the first column is the previous volatility

lasso3<- gamlr(spm2, y=V, lambda.min.ratio=1e-3)


# continue on your own
hist(lasso3$lambda)

plot(lasso3)

Betas <- drop(coef(lasso3))  # AICc default selection

length(Betas) # intercept product cats words

len_b <- length(Betas)

sum(Betas[251:len_b]!=0)  # predictive words

#choose 10 most positive review words in this model

o <- order(Betas[251:len_b],decreasing = T)

kable(Betas[251:len_b][o[1:10]])

# What is the in-sample R2 now

1- lasso3$deviance[which.min(AICc(lasso3))]/lasso3$deviance[1]

exp(Betas[names(Betas)=="terrorist"])
# each extra word discount increases the odds of 5 stars 1000 times
# the probability of 5 star when the review consists of just one word "terrorist" is
linpred<-as.numeric(Betas[1]+Betas[names(Betas)=="terrorist"])
kable(exp(linpred)/(1+exp(linpred))) # nearly 1
dev.off()

cv.fit <- cv.gamlr(spm,y=V,lambda.min.ratio=1e-3)

kable(cv.fit$lambda.min)

kable(cv.fit$lambda.1se) # larger -> fewer coefficients

Beta_cv1se <- coef(cv.fit) # 1se rule

kable(table(Beta_cv1se[,1]!=0)) # this includes the intercept

Beta_cvmin<-coef(cv.fit, select="min") # min cv selection

kable(table(Beta_cvmin[,1]!=0)) # a bit less strict, more nonzero coefficients

plot(cv.fit)

abline(v=log(lasso2$lambda[which.min(AICc(lasso2))]))






# Bootstrap to obtain s.e. of 1.s.e. chosen lambda


# We apply bootstrap to approximate
# the sampling distribution of lambda 
# selected by AICc

# export the data to the clusters 

Outcome<-V

clusterExport(cl,"spm2")
clusterExport(cl,"V")

# run 100 bootstrap resample fits

boot_function <- function(ib){

	require(gamlr)

	fit <- gamlr(spm2[ib,],y=V[ib], lambda.min.ratio=1e-3)

	fit$lambda[which.min(AICc(fit))]
}



boots <- 100

n <- nrow(spm2)

resamp <- as.data.frame(
			matrix(sample(1:n,boots*n,replace=TRUE),
			ncol=boots))

lambda_samp <- unlist(parLapply(cl,resamp,boot_function))

# continue on your own
summary(lambda_samp)
sd(lambda_samp)
lambda_p<-t.test(lambda_samp)
lambda_p$conf.int
hist(lambda_samp,col = "red")
print(lambda_p)
#***************************** 3 *******************************

# High-dimensional Covariate Adjustment 

d <- Previous # this is the treatment

# marginal effect of past on present volatility



summary(glm(V~d)) 

# we want to isolate the effect of d from external influences. We saw that words can explain some of the volatility.

x<-cbind(d,spm) # add the previous return to the model matrix
 




# Stage 1 LASSO: fit a model for d on x
naive <- gamlr(x,d)
head(coef(naive)["d",])
treat <- gamlr(x,d,lambda.min.ratio=1e-4)
# continue on your own

dhat <- predict(treat, x, type="response")
1- treat$deviance[which.min(AICc(treat))]/treat$deviance[1] #R2
plot(dhat,type="l")
# Stage 2 LASSO: fit a model for V using d, dhat and x
par(mfrow=c(1,2))
# continue on your own
causal <- gamlr(cBind(d,dhat,x),V,free = 2)
coef(causal)["d",]
1- causal$deviance[which.min(AICc(causal))]/causal$deviance[1] #R2
plot(causal)
naive <- gamlr(x,V)
coef(naive)["d",]
plot(naive)
