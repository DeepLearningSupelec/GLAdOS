%% Learning Curves of Convolution Neural Network
%Here we train the CNN and we show the accuracy curves when the training
%size grows
%======================================================================
%% Initializing Parameters and Loading Data

% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim = 9;    % Filter size for conv layer
numFilters = 20;   % Number of filters for conv layer
poolDim = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

%Initialization of parameters
theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses);

% Load MNIST Train
addpath ../common/;
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%Loading MNIST Test
testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%Setting number of Batch Test
nbTestBatch=1000;
batchTest=testImages(:,:,1:nbTestBatch);
batchTestLabels=testLabels(1:nbTestBatch);

%%======================================================================
%% Options
epochs = 1;
minibatch = 256;
alpha = 1e-1;
momentum = .95;               

%%======================================================================
%% Learning and plotting accuracy curves

% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));
m = length(labels);

%Learning by SGD
it = 0;
nbIterations=epochs*floor(size(images,3)/minibatch);
accuracies=zeros(1,nbIterations);
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = momentum;
        end;

        % get next randomly selected minibatch
        mb_data = images(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost grad] = cnnCost(theta,mb_data,mb_labels,numClasses,filterDim,numFilters,poolDim);
        
        % updating parameters using stochastic descent
        velocity = (mom.*velocity) + (alpha.*grad);
        theta = theta - velocity;
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        
%         % evaluating accuracy on training set
%         [~,~,trainPreds]=cnnCost(theta,testImages,testLabels,numClasses,...
%                 filterDim,numFilters,poolDim,true);
%         trainAcc = sum(trainPreds==labels)/length(trainPreds);
%         fprintf('      Accuracy is %f\n',trainAcc);
        
         % evaluating accuracy on testing set
        [~,~,testPreds]=cnnCost(theta,batchTest,batchTestLabels,numClasses,...
                filterDim,numFilters,poolDim,true);
            testAcc = sum(testPreds==batchTestLabels)/length(testPreds);
            fprintf('  Accuracy is %f\n',testAcc);
            accuracies(it)=testAcc;
    end;

    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;

end;

%Saving trained value of parameters
opttheta = theta;

%Plotting accuracy curve
figure
plot(minibatch*[1:nbIterations],accuracies*100)
xlabel('Training size')
ylabel('Accuracy % on batch test')

