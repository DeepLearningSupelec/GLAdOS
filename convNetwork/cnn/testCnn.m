%% Testing results
%This code is aimed to see how the CNN behaves when parameters are learned
%Therefore it should be launched only after the opttheta has been learned.


%% Setting the size of the batchtest

batchtest=100;
batchImages=testImages(:,:,1:batchtest);
batchLabels=testLabels(1:batchtest);

x=floor(sqrt(batchtest));
y=floor(batchtest/x)+1 ;
figure
for i=1:batchtest
    subplot(y,x,i)
    imshow(batchImages(:,:,i))
end
suptitle('Images de l''échantillon de test')

imageDim = size(batchImages,1); % height/width of image
numImages = size(batchImages,3); % number of images

%% Reshape parameter vector theta

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias

[Wc, Wd, bc, bd] = cnnParamsToStack(opttheta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

%%======================================================================

%% Show Learned Convoltution Filters

x1=floor(sqrt(numFilters));
y1=floor(numFilters/x1)+1 ;
figure
for i=1:numFilters
    subplot(y1,x1,i)
    imshow(Wc(:,:,i))
    str=sprintf('Filtre %d',i);
    title(str);
end
suptitle('Filtres du CNN')


%% Forward Propagation

%% Convolutional Layer

convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations

% activations = zeros(convDim,convDim,numFilters,numImages);
activations = cnnConvolve(filterDim, numFilters, batchImages, Wc, bc);
% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations

%Showing activations of an image as an example:
numImage=1; %Change this parameter to change the number of considered image
x1=floor(sqrt(numFilters));
y1=floor(numFilters/x1)+1 ;
figure
for i=1:numFilters
    subplot(y1,x1,i)
    imshow(activations(:,:,i,numImage))
    str=sprintf('Activation %d',i);
    title(str);
end
str1=sprintf('Activations de l''image numéro %d',numImage);
suptitle(str1)


% activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);
activationsPooled = cnnPool(poolDim, activations);


%Showing MeanPooled activations of an image as an example:

figure
for i=1:numFilters
    subplot(y1,x1,i)
    imshow(activationsPooled(:,:,i,numImage))
    str=sprintf('PooledActiv. %d',i);
    title(str);
end
str1=sprintf('Pooled Activations de l''image numéro %d',numImage);
suptitle(str1)

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer

% probs is a numClasses x numImages for storing probability that each image 
%belongs to each class.
probs = zeros(numClasses,numImages);

A=Wd*activationsPooled; %numClasses x numImages
%adding the bias
A=bsxfun(@plus,A,bd);
%preventing large values
A=bsxfun(@minus,A,max(A));
A=exp(A); %element_wise exponential
S=sum(A);
probs=bsxfun(@rdivide,A,S); %numClasses x numImages

%%======================================================================
%%  Calculate Cross Entropy Cost 
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%Construction of cost function J
I=sub2ind(size(probs), batchLabels',1:numImages);
values = log(probs(I));

%Storing result in cost
cost=-sum(values)/numImages; 

%% Calculate predictions
% Makes predictions given probs and returns without backproagating errors.

    [~,preds] = max(probs,[],1);
    preds = preds';
    preds(preds==10)=0;
 
% %Showing predictions 
% figure
% 
% for i=1:batchtest
%     subplot(y,x,i)
%     axis off
%     str=sprintf('%d',preds(i));
%     t = text(0.5,0.5,str);
%    s = t.FontSize;
%    t.FontSize = 12;
% 
% end
% suptitle('Prédiction des valeurs par le CNN')

%Showing and comparing labels and predictions


batchLabels(batchLabels==10)=0;
figure
for i=1:batchtest
    subplot(y,x,i)
    axis off
    
   if preds(i)==batchLabels(i)
       str=sprintf('%d',batchLabels(i));
    t = text(0.5,0.5,str);
   s = t.FontSize;
   t.FontSize = 12;
   t.Color='blue';
   else
       str=sprintf('%d|%d',batchLabels(i),preds(i));
    t = text(0.5,0.5,str);
   s = t.FontSize;
   t.FontSize = 12;
       t.Color='red';      
   end
end
suptitle('Labels des images | Predictions CNN')

%Showing probability distribution for mistaken predictions

Error=(preds~=batchLabels);
errorimages=find(Error);

for k=errorimages'
    prob=100*probs(:,k);
    copyprob=prob;
    prob(1)=copyprob(10);
    for i=1:9
        prob(i+1)=copyprob(i);
    end
    figure 
    imshow(batchImages(:,:,k))
    str=sprintf('Image numero %d', k);
    title(str)
     figure
    bar([0:9],prob')
    xlabel('Classes')
    ylabel('Probability %')
    str=sprintf('Probability distribution of image %d',k);
    title(str);

end

%% Calculating Accuracy
acc = sum(preds==batchLabels)/length(preds);
fprintf('Accuracy is %f\n',acc);

%%======================================================================