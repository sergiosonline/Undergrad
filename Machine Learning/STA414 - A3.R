###################
### STA414 - A3 ###
###################


### Data Preparation ###
## Preparing Training data
trainx <- read.csv("http://www.cs.toronto.edu/~rsalakhu/STA414_2015/train1x", 
                   sep=" ",header=F)
trainy <- read.csv("http://www.cs.toronto.edu/~rsalakhu/STA414_2015/train1y", 
                   sep=" ",header=F)
train1y <- 0
for(i in 1:length(trainy[[1]])){
  train1y <- c(train1y,as.numeric(trainy[[1]][i]))
}

(train1y <- train1y[2:251])

x1 <- trainx[,1]; x2 <- trainx[,2]; x3 <- trainx[,3]; x4 <- trainx[,4]; x5<- trainx[,5]; x6<- trainx[,6]; x7<- trainx[,7]; x8<- trainx[,8];

(train1x <- cbind(x1, x2, x3, x4, x5, x6, x7,x8))

# Training sets train1x and train1y now ready


## Preparing Testing data
testx <- read.csv("http://www.cs.toronto.edu/~rsalakhu/STA414_2015/testx",sep=" ",header=F)
testy <- read.csv("http://www.cs.toronto.edu/~rsalakhu/STA414_2015/testy",sep=" ",header=F)

(test1y <- testy[[1]])

ax1 <- testx[,1]; ax2 <- testx[,2]; ax3<- testx[,3]; ax4<- testx[,4]; ax5<- testx[,5]; ax6<- testx[,6]; ax7<- testx[,7]; ax8<- testx[,8]; 
test1x <- cbind(ax1, ax2, ax3, ax4, ax5, ax6, ax7,ax8)

testx <- 0; testy <- 0; trainx <- 0; trainy <-0

#Testing data's now ready

### 1. Linear model ###

### Training model
ptm <- proc.time()
mod0 <- lm(train1y ~ train1x)
summary(mod0)

### Making predictions
betas <- mod0$coefficients
pred0 <- test1x %*% betas[2:9]
(pred <- pred0 + betas[1])
pred[1]

### Evaluating predictions
sse0 <- sum((pred - test1y)^2);sse0 # SSE 719.1098
mse0 <- sse0/length(test1y); mse0
# MSE .2876
(proc.time() - ptm)


### GP with linear covariance ###

### Setting up noise-free covariance
# Training set train1x - 8 by 250
ptm <- proc.time()
K <- matrix(as.numeric(0),nrow=250,ncol=250)

for (z in 1:250){
  for (w in 1:250){
    K[z,w] <- sum(train1x[z,] * train1x[w,]) * 100^2 + as.numeric(z==w)
  }
}

invK <- solve(K) # Inverse

## Making predictions 
pred1 <- numeric(2500)

kk <- function(testindex){
  kk <- numeric(250)
  for (ll in 1:250){
    kk[ll] <- sum(train1x[ll,] * test1x[testindex,]) * 100^2
  }
  kk
}

k <- numeric(2500)
invKt <- invK %*% train1y
for(ii in 1:2500){
  k[ii] <- kk(ii) %*% invKt
}

## Prediction vector
k

sse1 <- sum((k-test1y)^2); sse1 #SSE is 1108.733
mse1 <- sse1/2500; mse1 # MSE is .4435
proc.time() - ptm



#### Fitting GP models #####

install.packages("parallel")
library(parallel)

# Note how new K is identical to MSE

# Crossvalidation 1

ctrx1 <- train1x[1:225,]; ctex1 <- train1x[226:250,]
ctry1 <- train1y[1:225]; ctey1 <- train1y[226:250]


### Creating crossvalidation data

CROSSX <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
for(i in 1:10){
  t <- i*25
  w <- ((i-1)*25)+1
  if (i < 2){
    CROSSX[[i]] <- train1x[i:t,]
  }
  CROSSX[[i]] <- train1x[w:t,]
}


CROSSY <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
for(i in 1:10){
  t <- i*25
  w <- ((i-1)*25)+1
  if (i < 2){
    CROSSY[[i]] <- train1y[i:t]
  }
  CROSSY[[i]] <- train1y[w:t]
}

assembler.x <- function(listf){
  final <- NA
  outtest <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
  outtrain <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
  for (j in 1:10){
    for (i in 1:10){
      if (i < j){
        outtrain[[j]] <- rbind(outtrain[[j]],listf[[i]])
        }
      if (i > j){
        outtrain[[j]] <- rbind(outtrain[[j]],listf[[i]])
      }
      outtest[[j]] <- listf[[j]]
      }
  }
  for (z in 1:10){
    outtrain[[z]] <- outtrain[[z]][2:226,]
  }
  
  final <- list(outtrain,outtest)
  final
}

assembler.y <- function(listf){
  final <- NA
  outtest <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
  outtrain <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA)
  for (j in 1:10){
    for (i in 1:10){
      if (i < j){
        outtrain[[j]] <- c(outtrain[[j]],listf[[i]])
      }
      if (i > j){
        outtrain[[j]] <- c(outtrain[[j]],listf[[i]])
      }
      outtest[[j]] <- listf[[j]]
    }
  }
  for (z in 1:10){
    outtrain[[z]] <- outtrain[[z]][2:226]
  }
  
  final <- list(outtrain,outtest)
  final
}

# First training set, excluding obs 1-25: assembler(CROSSX)[[1]][[1]]



#train.CROSSX.2 <- rbind(CROSSX[[1]],CROSSX[[3]],CROSSX[[4]],CROSSX[[5]],CROSSX[[6]],CROSSX[[7]],CROSSX[[8]],CROSSX[[9]]
#                        ,CROSSX[[10]])
#train.CROSSY.2 <- c(CROSSY[[1]],CROSSY[[3]],CROSSY[[4]],CROSSY[[5]],CROSSY[[6]],CROSSY[[7]],CROSSY[[8]],CROSSY[[9]]
#                    ,CROSSY[[10]])
#test.CROSSX.2 <- CROSSX[[2]]
#test.CROSSY.2 <- CROSSY[[2]]

# Use rbind to put together training datasets



### Creating vector of potential hyperparameters

gamt <- numeric(20)
rhot <- numeric(20)

for (i in 1:20){
  gamt[i] <- gamt[i] + i*.5
}
(gamt <- c(.1,gamt[1:19]+.1,gamt[20]))

for(j in 1:20){
  rhot[j] <- rhot[j]+.05*j
}
(rhot <- c(.01,rhot[1:19]+.01,rhot[20]))


# Basic functions

K.basic <- function (x1,x2,r,g){
  100^2 + g^2 * exp(-r^2 * sum((x1-x2)^2))
}


K.train <- function (xt, x, rh, gam){
  
  K3 <- matrix(0,nrow(xt),nrow(x))
  
  for(i in 1:nrow(xt) ){
    for (j in 1:nrow(x)){
      K3[i,j] <- K.basic(xt[i,],x[j,],rh,gam) + as.numeric(i==j)
    }
  }
  K3
}

# Testing K builder
# K.train(ctrx1,ctrx1,rhot,gamt)

pred1 <- function(trx1,trx2,try1,rho,gamma){
  
  predd <- numeric(25)
  Kmed <- K.train(trx1,trx1,rho,gamma)
  invKmed <- solve(Kmed)
  invKmedt <- invKmed %*% try1
  
  for(jj in 1:25){
    ip <- numeric(225)
    for(ee in 1:225){
      ip[ee] <- K.basic(trx1[ee,],trx2[jj,],rho,gamma)
    }
    predd[jj] <- ip %*% invKmedt 
  }
  predd
}

# Putting together cross validation sets


#### Final calculations

mscal_ext <- function(X,Y,R,G){
  # X is matrix of observations and covariates. dim(X)= NxD
  # Y is vector of dependent variables. dim(Y)=Nx1
  # R is vector of rhos
  # G is vector of gammas
  matx <- assembler.x(X) # List of 2 lists: (list of training matrices, list of corresponding test matrices)
  maty <- assembler.y(Y) # List of 2 lists: (list of training matrices, list of corresponding test matrices)
  holder <- numeric(10)
  HH <- matrix(numeric(441),nrow=21,ncol=21)
  
  for (a in 1:length(R)){
    for (b in 1:length(G)){
      print(c("gamma = ", G[b], " #### rho = ", R[a]))
      for(c in 1:10){
        print(c)
          holder[c] <- (sum((pred1(matx[[1]][[c]],matx[[2]][[c]],maty[[1]][[c]],R[a],G[b]) -maty[[2]][[c]])^2))/25
        }
    HH[a,b] <- mean(holder)
    holder <- numeric(10)
      }
  }
  HH
}


#sum((pred1(ctrx1,ctex1,ctry1,rhot[1],gamt[1]) - ctey1)^2) # MSE = .4798

# Testing
#train.CROSSX.1 <- rbind(CROSSX[[2]],CROSSX[[3]],CROSSX[[4]],CROSSX[[5]],CROSSX[[6]],CROSSX[[7]],CROSSX[[8]],CROSSX[[9]]
#                        ,CROSSX[[10]])
#train.CROSSY.1 <- c(CROSSY[[2]],CROSSY[[3]],CROSSY[[4]],CROSSY[[5]],CROSSY[[6]],CROSSY[[7]],CROSSY[[8]],CROSSY[[9]]
#                        ,CROSSY[[10]])
#test.CROSSX.1 <- CROSSX[[1]]
#test.CROSSY.1 <- CROSSY[[1]]

#mscalc(train.CROSSX.1,test.CROSSX.1,train.CROSSY.1,test.CROSSY.1,rhot,gamt)

### Result ###
rhot
gamt

ptm <- proc.time()
BAMFM <- mscal_ext(CROSSX,CROSSY,rhot,gamt); BAMFM # Cross validation matrix
# Optimal params: gamma 10, rho .16, MSE=.23488 

# Generating MSE from predictions generated by optimal hyperparams

pred2 <- function(trx1,trx2,try1,rho,gamma){
  
  predd <- numeric(25)
  Kmed <- K.train(trx1,trx1,rho,gamma)
  invKmed <- solve(Kmed)
  invKmedt <- invKmed %*% try1
  
  for(jj in 1:2500){
    ip <- numeric(250)
    for(ee in 1:250){
      ip[ee] <- K.basic(trx1[ee,],trx2[jj,],rho,gamma)
    }
    predd[jj] <- ip %*% invKmedt 
  }
  predd
}


pred_final1 <- pred2(train1x,test1x,train1y,rhot[4],gamt[21]);pred_final1
ssegp1 <- sum((pred_final1-test1y)^2); ssegp1
msegp1 <- ssegp1/2500; msegp1
proc.time() - ptm

# Export BAMFM to Excel
write.csv(BAMFM,file="BAMFM.csv")

### Reconsidering all 3 methods for the transformed dataset

# Divide covariates 1 and 7 by 10
train1x;test1x
train1y;test1y

newtrain1x <- cbind(train1x[,1]/10,train1x[,2:6],train1x[,7]/10,train1x[,8])
newtest1x <- cbind(test1x[,1]/10,test1x[,2:6],test1x[,7]/10,test1x[,8])

CROSSX2 <- list(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA) 
for(i in 1:10){
  t <- i*25
  w <- ((i-1)*25)+1
  if (i < 2){
    CROSSX2[[i]] <- newtrain1x[i:t,]
  }
  CROSSX2[[i]] <- newtrain1x[w:t,]
}


### Linear model 2###

### Training model
ptm <- proc.time()
mod1 <- lm(train1y ~ newtrain1x)
summary(mod1)

### Making predictions
betas1 <- mod1$coefficients
pred1 <- newtest1x %*% betas1[2:9]
pred2 <- pred1 + betas1[1]

### Evaluating predictions
sse1 <- sum((pred2 - test1y)^2);sse1 # 
mse1 <- sse1/length(test1y); mse1
# 
proc.time() - ptm


### GP with linear covariance 2 ###

### Setting up noise-free covariance
# Training set newtrain1x - 8 by 250
ptm <- proc.time()
Kale <- matrix(as.numeric(0),nrow=250,ncol=250)

for (z in 1:250){
  for (w in 1:250){
    Kale[z,w] <- sum(newtrain1x[z,] * newtrain1x[w,]) * 100^2 + as.numeric(z==w)
  }
}

invKale <- solve(Kale) # Inverse

### Making predictions 
pred_linGP1 <- numeric(2500)


kk2 <- function(testindex){
  kk <- numeric(250)
  for (ll in 1:250){
    kk[ll] <- sum(newtrain1x[ll,] * newtest1x[testindex,]) * 100^2
  }
  kk
}


kaley <- numeric(2500)
invKaleyt <- invKale %*% train1y
for(ii in 1:2500){
  kaley[ii] <- kk2(ii) %*% invKaleyt
}

### Prediction vector
kaley

sselingp1 <- sum((kaley-test1y)^2); sselingp1 #SSE is 1108.733
mselingp1 <- sselingp1/2500; mselingp1 # MSE is .4435
proc.time() - ptm


### General GP 2 ###

ptm <- proc.time()

BAMFM2 <- mscal_ext(CROSSX2,CROSSY,rhot,gamt);BAMFM2 # Cross validation matrix
### Optimal params gamma 4.1, rho 1, MSE=.28309

pred_final2 <- pred2(newtrain1x,newtest1x,train1y,rhot[21],gamt[9]);pred_final2
ssegp2 <- sum((pred_final2-test1y)^2); ssegp2
msegp2 <- ssegp2/2500; msegp2

proc.time() - ptm

write.csv(BAMFM2,file="BAMFM2.csv")