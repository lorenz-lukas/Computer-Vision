%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Classification comparison between Knn, Qda, Lda and Baysian classifier. 
%% Lukas Lorenz de Andrade - 16/0135109
%% Gustavo                 - 
%%  
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

X_training = X(1:70);
Y_training = Y(1:70);
X_sample = X(71:100);
Y_sample = Y(71:100);

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
% Validation
Y_predicted = predict(Mdl,X_sample);
[C,order] = confusionmat(Y_sample,Y_predicted,'Order',{'1','2','3','4'})
accuracy=(C(1,1)+C(2,2)+C(3,3))/sum(sum(C))

%%%% Baysian classifier:
 
 
 
 


%% Open a new image to test the classifiers
img = imread('photo-9-orig.jpg');
img = imresize(img,6);
figure, imshow(img);
[x,y] = ginput(1);
px = img(int16(x),int16(y));

%%%% Knn output:
display('Your Knn prediction is:');
flwrClass = predict(Mdl,px)