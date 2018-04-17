%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Classification comparison between Knn, Qda, Lda and Baysian classifier. 
%%% Lukas Lorenz de Andrade - 16/0135109
%%% Gustavo                 - 
%%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dados = xlsread('dataset.xlsx');
pixel = dados(1,1:100); 
label = dados(2,1:100);

%%% Randomizing data
rng(0,'twister'); % For reproducibility
numObs = length(pixel);
p = randperm(numObs);
pixel = pixel(p,:);
label = label(p);
X = pixel;
Y = label; 
rng(1); % For reproducibility

X_training = X(1:100);
Y_training = Y(1:100);

%%%% Knn:
%k = input('Set the k value:  ');
k = 10;
Mdl = fitcknn(X_training,Y_training,'NumNeighbors',k);
display('ERRORS:')
rloss = resubLoss(Mdl)
CVMdl = crossval(Mdl);
kloss = kfoldLoss(CVMdl)
    
display('OBS: rloss -> missclassification fraction')
display('kloss ->  average loss of each cross-validation model when predicting on new data.')

%%%% Baysian classifier:
nbGau = fitcnb(meas(:,1:2), species);
nbGauResubErr = resubLoss(nbGau)
labels = predict(nbGau, [x y]); 

%melhorando
nbKD = fitcnb(meas(:,1:2), species, 'DistributionNames','kernel', 'Kernel','box');
nbKDResubErr = resubLoss(nbKD)
labels = predict(nbKD, [x y]);
%%%% LDA
lda = fitcdiscr(X_training,Y_training);
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda)
[ldaResubCM,grpOrder] = confusionmat(species,ldaClass)

%%%% QDA
qda = fitcdiscr(meas(:,1:2),species,'DiscrimType','quadratic');
qdaResubErr = resubLoss(qda)
%%% Open a new image to test the classifiers
photo = input('PHOTOs NAME:')
img = imread(photo);
img = imresize(img,6);
[m,n] = size(img)

for j = 1 : (m*n)
    append(Y_predicted,predict(Mdl,X_sample));
end
    
    [C,order] = confusionmat(Y_sample,Y_predicted,'Order',{'1','2','3','4'})
    accuracy=(C(1,1)+C(2,2)+C(3,3))/sum(sum(C))
    
    

%%%% Knn output:

flwrClass = predict(Mdl,px)
