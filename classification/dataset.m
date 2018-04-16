%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Creating dataset to classification project based on given images. 
%% Lukas Lorenz de Andrade - 16/0135109
%% Gustavo                 - 
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img = imread('photo-9-orig.jpg');
img = imresize(img,6);
figure, imshow(img);
for i = 1:100 
    [y,x] = ginput(1);
    px = img(int16(x),int16(y),:);
    fprintf('%d input\n\n', i);
    label = input('Set the pixel label according to:\n1.Green leaves; 2. Ground;\n3. Yellow and red leaves and fruits; 4. Shadows or unknown.\n')
    data(i,1) = px(:,:,1);
    data(i,2) = px(:,:,2);
    data(i,3) = px(:,:,3);
    data(i,4) = label;
end
xlswrite('dataset.xlsx',data);
close all
%dados = xlsread('pendulo_trifilar_pequeno_ugollini.xlsx');
%tempo1 = dados(1:3323,1); % 1:3323
