% Project 1: Knn algorithm applied to Fisher Iris dataset
% Dataset information:
%       - Number of atributes: 4 (sepal and petal lenght and width);
%       - Number of classes: 3 (Iris Setosa, Iris Versicolour, Iris
%       Virginica);
%       - Total of samples: 150;
% Lukas Lorenz de Andrade
% References: 
%   http://www.mathworks.com/help/stats/classification-using-nearest-neighbors.html#btap7k2
%   https://www.mathworks.com/help/stats/confusionmat.html
%   https://www.mathworks.com/matlabcentral/fileexchange/37827-4-nearest-neighbor-on-iris-recognition-using-randomized-partitioning?focused=5243431&tab=function
close all
load fisheriris

rng(0,'twister'); % For reproducibility
numObs = length(species);
p = randperm(numObs);
meas = meas(p,:);
species = species(p);
X = meas;
Y = species; 
rng(1); % For reproducibility

hold on;
scatter(X(:,1),X(:,2),'+','r')
scatter(X(:,3),X(:,4),'k*')
title ('Fisher''s Iris Data');
xlabel ('Lengt (cm)'); 
ylabel ('Widths (cm)');
legend('Sepal','Petal');
grid
hold off

x(1) = input('Set the Iris sepal lenght(cm):  ');
x(2) = input('Set the Iris sepal windth(cm):  ');
x(3) = input('Set the Iris petal lenght(cm):  ');
x(4) = input('Set the Iris petal windth(cm):  ');

close all

X_training = X(1:100,:);
Y_training = Y(1:100);
X_sample = X(101:150,:);
Y_sample = Y(101:150);

%for i = 1:150
for i = 1:5
    k = input('Set the k value:  ');
    %Mdl = fitcknn(X_training,Y_training,'NumNeighbors',i);
    Mdl = fitcknn(X_training,Y_training,'NumNeighbors',k);
    
    display('ERRORS:')
    %rloss(:,i) = resubLoss(Mdl);
    rloss = resubLoss(Mdl)
    CVMdl = crossval(Mdl);
    %kloss = kfoldLoss(CVMdl);
    kloss = kfoldLoss(CVMdl)
    
    display('OBS: rloss -> missclassification fraction')
    display('kloss ->  average loss of each cross-validation model when predicting on new data.')
    
    Y_predicted = predict(Mdl,X_sample);
    [C,order] = confusionmat(Y_sample,Y_predicted,'Order',{'setosa','versicolor','virginica'})
    accuracy=(C(1,1)+C(2,2)+C(3,3))/sum(sum(C))
    
end
display('Your flower classification is:');
flwrClass = predict(Mdl,x)
% To generate the error plot, uncomment the lines in the loop and the ones bellow and comment
% the other ones.
%hold on;
%plot(1:150,rloss)
%plot(16,rloss(16),'*')
%title ('Knn applied to Iris');
%xlabel ('K number'); 
%ylabel ('Error');
%grid
%hold off 