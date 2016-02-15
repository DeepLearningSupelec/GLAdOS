function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

for imageNum = 1:numImages
  for filterNum = 1:numFilters

    % convolution of image with feature matrix
    convolvedImage = zeros(convDim, convDim);

    % Obtaining the feature (filterDim x filterDim) needed during the convolution

    filter=W(:,:,filterNum);

    % Flipping the feature matrix because of the definition of convolution
    filter = rot90(squeeze(filter),2);
      
    % Obtaining the image
    im = squeeze(images(:, :, imageNum));

    % Convolving "filter" with "im" and adding the result to convolvedImage
    convolvedFeature=conv2(im,filter,'valid');
    convolvedImage=convolvedImage+convolvedFeature;
    % Adding the bias unit
    bias=squeeze(b(filterNum,1));
    convolvedImage=convolvedImage+bias;
    % Applying the sigmoid function to get the hidden activation
    convolvedImage=sigmf(convolvedImage,[1 0]);
    
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end


end

