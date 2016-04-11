function [curColumnMap] = returnCAMmap( featureObjectSwitchSpatial, weights_LR)
%RETURNCOLUMNMAP Summary of this function goes here
%   Detailed explanation goes here

if size(featureObjectSwitchSpatial,4) ==1
    
    featureObjectSwitchSpatial_vectorized = reshape(featureObjectSwitchSpatial,[size(featureObjectSwitchSpatial,1)*size(featureObjectSwitchSpatial,2) size(featureObjectSwitchSpatial,3)]);
    detectionMap = featureObjectSwitchSpatial_vectorized*weights_LR;
    curColumnMap = reshape(detectionMap,[size(featureObjectSwitchSpatial,1),size(featureObjectSwitchSpatial,2), size(weights_LR,2)]);
else
    columnSet = zeros(size(featureObjectSwitchSpatial,1),size(featureObjectSwitchSpatial,2),size(weights_LR,2),size(featureObjectSwitchSpatial,4));
    for i=1:size(featureObjectSwitchSpatial,4)
        curFeatureObjectSwitchSpatial = squeeze(featureObjectSwitchSpatial(:,:,:,i));
        featureObjectSwitchSpatial_vectorized = reshape(curFeatureObjectSwitchSpatial,[size(curFeatureObjectSwitchSpatial,1)*size(curFeatureObjectSwitchSpatial,2) size(curFeatureObjectSwitchSpatial,3)]);
        detectionMap = featureObjectSwitchSpatial_vectorized*weights_LR;
        curColumnMap = reshape(detectionMap,[size(featureObjectSwitchSpatial,1),size(featureObjectSwitchSpatial,2), size(weights_LR,2)]);
        columnSet(:,:,:,i) = curColumnMap;
    end
    curColumnMap = columnSet;
end

   

end

