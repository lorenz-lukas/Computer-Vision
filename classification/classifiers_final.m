%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification comparison between Knn, Qda, Lda and Baysian classifier. 
%%% Lukas Lorenz de Andrade - 16/0135109
%%% Gustavo                 - 
%%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dados = xlsread('dataset.xlsx');
X_training = dados(1:100,1:3); 
Y_training = dados(1:100,4);

%%% Randomizing data
%rng(0,'twister'); % For reproducibility
%numObs = length(pixel);
%p = randperm(numObs);
%pixel = pixel(p,:);
%label = label(p);
%X = pixel;
%Y = label; 
%rng(1); % For reproducibility

%%%% Knn:
%k = input('Set the k value:  ');
k = 8;
Mdl = fitcknn(X_training,Y_training,'NumNeighbors',k);
display('ERRORS:')
rloss = resubLoss(Mdl)
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)
    
display('OBS: rloss -> missclassification fraction')
display('kloss ->  average loss of each cross-validation model when predicting on new data.')

%%%% Baysian classifier:
nbGau = fitcnb(X_training, Y_training);
nbGauResubErr = resubLoss(nbGau)

%%%% LDA
lda = fitcdiscr(X_training,Y_training);
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda)
[ldaResubCM,grpOrder] = confusionmat(Y_training,ldaClass)

%%%% QDA
qda = fitcdiscr(X_training,Y_training, 'DiscrimType','quadratic');
qdaClass = resubPredict(qda);
qdaResubErr = resubLoss(qda)
[qdaResubCM,grpOrder] = confusionmat(Y_training,qdaClass)
%%% Open a new image to test the classifiers
img = imread('photo-1-orig.jpg');
%px = img(i,j,:);
%rgb(j,1) = px(:,:,1);
%rgb(j,2) = px(:,:,2);
%rgb(j,3) = px(:,:,3);
img = xlsread('DADOS1.xlsx');
%%%% KNN output
Y_predicted = predict(Mdl,X_training);
[C,order] = confusionmat(Y_training, Y_predicted,'Order',{'R1','R2','R3','R4'})
accuracy=(C(1,1)+C(2,2)+C(3,3))/sum(sum(C))
%%%% Baysian 
%%%%%%%%%%%%%%%% NAIVE
Y_predicted = predict(nbGau, img); 
%melhorando
nbKD = fitcnb(X_training, Y_training, 'DistributionNames','kernel', 'Kernel','box');
nbKDResubErr = resubLoss(nbKD)
labels = predict(nbKD, img);
%%%% LDA
%classify([x y],meas(:,1:2),species);
%%%% QDA
%classify([x y],meas(:,1:2),species);
