% Sample code to generate class activation map from 10 crops of activations
% Bolei Zhou, March 15, 2016
% for the online prediction, make sure you have complied matcaffe

clear
addpath('/opt/caffe/matlab');

imgID = 2; % 1 or 2
img = imread(['img' num2str(imgID) '.jpg']);

img = imresize(img, [256 256]);
online = 0; % whether extract features online or load pre-extracted features

load('categories1000.mat');
if online == 1
    % load the CAM model and extract features

    net_weights = ['models/imagenet_googlenetCAM_train_iter_120000.caffemodel'];
    net_model = ['models/deploy_googlenetCAM_imagenet.prototxt'];
    net = caffe.Net(net_model, net_weights, 'test');    
    
    weights_LR = net.params('CAM_fc',1).get_data();% get the softmax layer of the network
   
    scores = net.forward({prepare_image(img)});% extract conv features online
    activation_lastconv = net.blobs('CAM_conv').get_data();
	scores = scores{1};
else
    % use the extracted features and softmax parameters cached before hand
    load('data_net.mat'); % it contains the softmax weights and the category names of the network
    load(['data_img' num2str(imgID) '.mat']); %it contains the pre-extracted conv features
end




%% Class Activation Mapping

topNum = 5; % generate heatmap for top X prediction results
scoresMean = mean(scores,2);
[value_category, IDX_category] = sort(scoresMean,'descend');
[curCAMmapAll] = returnCAMmap(activation_lastconv, weights_LR(:,IDX_category(1:topNum)));

curResult = im2double(img);
curPrediction = '';

for j=1:topNum
    curCAMmap_crops = squeeze(curCAMmapAll(:,:,j,:));
    curCAMmapLarge_crops = imresize(curCAMmap_crops,[224 224]);
    curCAMmap_image = mergeTenCrop(curCAMmapLarge_crops);

    curHeatMap = map2jpg(curCAMmap_image, [], 'jet');
    curHeatMap = im2double(img)*0.2+curHeatMap*0.7;
    curResult = [curResult ones(size(curHeatMap,1),8,3) curHeatMap];
    curPrediction = [curPrediction ' --top'  num2str(j) ':' categories{IDX_category(j)}];
    
end
disp(curPrediction);
imwrite(curResult, 'result.jpg');
figure,imshow(curResult);
title(curPrediction)

if online==1
    caffe.reset_all();
end

