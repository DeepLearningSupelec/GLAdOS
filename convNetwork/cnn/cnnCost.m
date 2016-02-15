function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias

[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% Forward Propagation

%% Convolutional Layer

convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations

% activations = zeros(convDim,convDim,numFilters,numImages);
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations

% activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);
activationsPooled = cnnPool(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer

% numClasses x numImages for storing probability that each image belongs to
% each class.
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
%% Calculating Cost


cost = 0; % save objective into cost


%Construction of cost function J
I=sub2ind(size(probs), labels',1:numImages);
values = log(probs(I));

%Storing result in cost
cost=-sum(values)/numImages; 

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% Backpropagation

%Backpropagation Softmax Layer

errorsSoftmax=probs;
errorsSoftmax(I)=errorsSoftmax(I)-1;
errorsSoftmax=errorsSoftmax/numImages; %numClasses x numImages
%Wd is %numClasses x hiddenSize
%Backpropatiaton Pooling Layer
errorsPooled=Wd'*errorsSoftmax; %hiddenSize x numImages
errorsPooled = reshape(errorsPooled, [], outputDim, numFilters, numImages);

%Backpropagation through subsampling

upsamplederrors=zeros(convDim,convDim,numFilters,numImages);
%upsampling the errors 
for imageNum=1:numImages
    for filterNum=1:numFilters
        delta=errorsPooled(:,:,filterNum,imageNum);
        upsamplederrors(:,:,filterNum,imageNum)=(1/poolDim^2) * kron(delta,ones(poolDim));
    end
end
%Backpropagation Conv Layer

%We choose to use the sigmoid as activation function

errorsConv=upsamplederrors.*activations.*(1-activations);

% Dimension check up :
%   - upsamplederrors is   convDim*convDim*numFilters*numImages
%   - activations is       convDim*convDim*numFilters*numImages
%   - errorsConv is        convDim*convDim*numFilters*numImages as expected

%%======================================================================
%%  Gradient Calculation


%=======================================%
%First we will compute the gradients for the Softmax layer
% This is done by the same algorithm as in a usual NN network 

%Softmax bias gradients
bd_grad=sum(errorsSoftmax,2); %notice that this is the sum of gradients.

%Softmax weights gradients
Wd_grad=errorsSoftmax*activationsPooled';

%Dimension Checkup
%   - errorSoftmax is      numClasses x numImages
%   - activationPooled' is numImages x hiddenSize
%   - Wd_grad is           numClassesxhiddenSize as expected
   
   

%=======================================%

%Here, We compute the gradients for the ConvLayer

%Filters bias gradients
for filterNum=1:numFilters
    %let delta be the numImages convDim*convDim error matrices for the 
    %considered filter
    delta=squeeze(errorsConv(:,:,filterNum,:));
    bc_grad(filterNum)=sum(delta(:));
end

%Filters weights gradients
for filterNum=1:numFilters
    %We initialise Wc_gradFilter that will contain the sum gradient of the
    %considered filter in the loop over all the images
    gradFilter=zeros(filterDim,filterDim);
    for imageNum=1:numImages
        %We convolve the backpropagated error with the considered filter
     errorConv=errorsConv(:,:,filterNum,imageNum);   
     errorConv=rot90(errorConv,2);
     %Sum of gradients over all images for considered filter
     gradFilter=gradFilter+conv2(images(:,:,imageNum),errorConv,'valid');
    end
    %Save value in Wc_grad before taking the next filter
    Wc_grad(:,:,filterNum)=gradFilter;
end

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
