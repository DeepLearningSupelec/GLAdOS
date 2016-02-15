function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, convolvedDim / poolDim, numFilters, numImages);


%  We use mean pooling here.

for imageNum = 1:numImages
  for filterNum = 1:numFilters

    %Initializing pooling filter
    poolingfilter=ones(poolDim,poolDim);
    % Obtaining the convolvedFeature for fixed imageNum and filterNum
    convolvedFeature = squeeze(convolvedFeatures(:, :, filterNum,imageNum));
    %pooledFeature contains the result of the pooled convolvedFeature
    pooledFeature=conv2(convolvedFeature,poolingfilter,'valid');
    %Subsampling
    pooledFeature=pooledFeature(1:poolDim:end,1:poolDim:end);
    %Averaging
    pooledFeature=pooledFeature/(poolDim*poolDim);
    %Saving the result in pooledFeature
    pooledFeatures(:, :, filterNum, imageNum) = pooledFeature;
  end
end

end

