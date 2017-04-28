function alignImgMean = mergeTenCrop( CAMmap_crops)
% align the ten crops of CAMmaps back to one image (take a look at caffe
% matlab wrapper about how ten crops are generated)
cropSize = size(CAMmap_crops,1);
cropImgSet = zeros([cropSize cropSize 3 10]);
cropImgSet(:,:,1,:) = CAMmap_crops;
cropImgSet(:,:,2,:) = CAMmap_crops;
cropImgSet(:,:,3,:) = CAMmap_crops;

    
squareSize = 256;
indices = [0 squareSize-cropSize] + 1;

alignImgSet = zeros(squareSize, squareSize, size(cropImgSet,3),'single');


curr = 1;
for i = indices
  for j = indices

    curCrop1 = permute(cropImgSet(:,:,:,curr),[2 1 3 4]);
    curCrop2 = permute(cropImgSet(end:-1:1,:,:,curr+5),[2 1 3 4]);


    alignImgSet(j:j+cropSize-1, i:i+cropSize-1,:,curr) = curCrop1;
    alignImgSet(j:j+cropSize-1, i:i+cropSize-1,:, curr+5) = curCrop2;
    
    curr = curr + 1;

  end
end
center = floor(indices(2) / 2)+1;
curCrop1 = permute(cropImgSet(:,:,:,5),[2 1 3 4]);
curCrop2 = permute(cropImgSet(end:-1:1,:,:,10),[2 1 3 4]);
alignImgSet(center:center+cropSize-1, center:center+cropSize-1,:,5) = curCrop1;
alignImgSet(center:center+cropSize-1, center:center+cropSize-1,:, 10) = curCrop2;
alignImgMean = squeeze(sum(sum(abs(alignImgSet),3),4));
alignImgMean = im2double(alignImgMean);
end



