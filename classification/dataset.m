%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Creating dataset to classification project based on given images. 
%% Lukas Lorenz de Andrade - 16/0135109
%% Gustavo Costa           - 14/0142568 
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

img= imread('photo-1-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_1 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-2-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_2 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-3-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_3 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-4-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_4 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-5-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_5 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-6-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_6 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-7-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_7 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-8-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_8 = [imgNew1,imgNew2,imgNew3];

img= imread('photo-10-orig.jpg');
imgNew1 = img(:,1,1);
imgNew2 = img(:,1,2);
imgNew3 = img(:,1,3);
for i=1:124
    imgNew1 = [imgNew1;img(:,i+1,1)];
end
for i=1:124
    imgNew2 = [imgNew2;img(:,i+1,2)];
end
for i=1:124
    imgNew3 = [imgNew3;img(:,i+1,3)];
end
imgNew_10 = [imgNew1,imgNew2,imgNew3];

imgNew = [imgNew_1; imgNew_2; imgNew_3; imgNew_4; imgNew_5; imgNew_6; imgNew_7; imgNew_8; imgNew_10];

xlswrite('DADOS9_images.xlsx', imgNew);

clear imgNew1 imgNew2 imgNew3 imgNew_1 imgNew_2 imgNew_3 imgNew_4 imgNew_5 imgNew_6 imgNew_7 imgNew_8 imgNew_10

close all