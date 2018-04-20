%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification comparison between Knn, Qda, Lda and Baysian classifier. 
%%% Lukas Lorenz de Andrade - 16/0135109
%%% Gustavo Costa           - 14/0142568 
%%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dados = xlsread('dataset.xlsx');
pixel = dados(1:100, 1:3);
label = dados(1:100, 4);
%%% Randomizing data
rng(0,'twister'); % For reproducibility
numObs = length(dados(1:100, 4));
p = randperm(numObs);
pixel = pixel(p,:);
label = label(p);
X = pixel;
Y = label; 
rng(1); % For reproducibility

X_training = pixel(1:70,1:3); 
Y_training = label(1:70);
X_test = pixel(71:100, 1:3);
Y_test = label(71:100);

DADOS = xlsread('DADOS9_images.xlsx');


%%%% Knn:
%k = input('Set the k value:  ');
k = 5;
Mdl = fitcknn(X_training,Y_training,'NumNeighbors',k);
display('ERRORS:')
rloss = resubLoss(Mdl)
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)

Y_predicted = predict(Mdl,X_test);
[C,order] = confusionmat(Y_test, Y_predicted)
accuracy=(C(1,1)+C(2,2)+C(3,3))/sum(sum(C))

Y_predicted_KNN = predict(Mdl, DADOS);
    
display('OBS: rloss -> missclassification fraction')
display('kloss ->  average loss of each cross-validation model when predicting on new data.')

%%%% Baysian classifier:
nbGau = fitcnb(X_training, Y_training);
nbGauResubErr = resubLoss(nbGau)

Y_predicted = predict(nbGau, X_test); 
[Teste, nbaGauResubErr] = confusionmat(Y_test,Y_predicted)

Y_predicted_BAYES = predict(nbGau, DADOS);
%melhorando
nbKD = fitcnb(X_training, Y_training, 'DistributionNames','kernel', 'Kernel','box');
nbKDResubErr = resubLoss(nbKD)
labels = predict(nbKD, X_test);

%%%% LDA
lda = fitcdiscr(X,Y);
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda)
[ldaResubCM,grpOrder] = confusionmat(Y,ldaClass)

%%%% QDA
qda = fitcdiscr(X,Y, 'DiscrimType','quadratic');
qdaClass = resubPredict(qda);
qdaResubErr = resubLoss(qda)
[qdaResubCM,grpOrder] = confusionmat(Y,qdaClass)

%%% Open a new image to test the classifiers
%img = imread('photo-1-orig.jpg');
%img = xlsread('DADOS1.xlsx');
%%%% KNN output

%%%% Baysian 
%%%%%%%%%%%%%%%% NAIVE

%%%% LDA
%classify([x y],meas(:,1:2),species);
%%%% QDA
%classify([x y],meas(:,1:2),species);