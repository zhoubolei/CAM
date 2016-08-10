% raw script used to generate heatmaps for ILSVRC localization experiment
% please load the necessary packages like matcaffe and ILSVRC toolbox correctly, some functions in matcaffe might be already deprecated.
% you could take it as an example to see how to reproduce the ILSVRC localization experiment.
%
% Bolei Zhou. 

addpath('caffeCPU2/matlab/caffe');

modelSetFolder = 'CAMnet';

%% CAMnet 


% netName = 'CAM_googlenetBVLC_imagenet';
% model_file = [modelSetFolder '/googlenet_imagenet/bvlc_googlenet.caffemodel'];
% model_def_file = [modelSetFolder '/googlenet_imagenet/deploy.protxt'];

% netName = 'CAM_alexnet';
% model_file = [modelSetFolder '/alexnet/CAMmodels/caffeNetCAM_imagenet_train_iter_100000.caffemodel'];
% model_def_file = [modelSetFolder '/alexnet/deploy_caffeNetCAM.prototxt'];

netName = 'CAM_googlenetBVLCshrink_imagenet';
model_file = [modelSetFolder '/googlenet_imagenet/CAMmodels/imagenet_googleletCAM_train_iter_80000.caffemodel'];
model_def_file = [modelSetFolder '/googlenet_imagenet/deploy_googlenetCAM.prototxt'];


% netName = 'CAM_VGG16_imagenet';
% model_file = [modelSetFolder '/VGGnet/models/vgg16CAM_train_iter_50000.caffemodel'];
% model_def_file = [modelSetFolder '/VGGnet/deploy_vgg16CAM.prototxt'];


%% loading the network
caffe('init', model_def_file, model_file,'test');
caffe('set_mode_gpu');
caffe('set_device',0);

%% testing to predict some image

weights = caffe('get_weights');
weights_LR = squeeze(weights(end,1).weights{1,1});
bias_LR = weights(end,1).weights{2,1};
layernames = caffe('get_names');
response = caffe('get_all_layers');
netInfo = cell(size(layernames,1),3);
for i=1:size(layernames,1)
    netInfo{i,1} = layernames{i};
    netInfo{i,2} = i;
    netInfo{i,3} = size(response{i,1});
end

load('categoriesImageNet.mat');
d = load('/data/vision/torralba/small-projects/bolei_deep/caffe/ilsvrc_2012_mean.mat');
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 256;
CROPPED_DIM = netInfo{1,3}(1); 

weightInfo = cell(size(weights,1),1);
for i=1:size(weights,1)
    weightInfo{i,1} = weights(i,1).layer_names;
    weightInfo{i,2} = weights(i,1).weights{1,1};
    weightInfo{i,3} = size(weights(i,1).weights{1,1});
end

%% testing to predict some image

datasetName = 'ILSVRCvalSet';
datasetPath = '/data/vision/torralba/gigaSUN/deeplearning/dataset/ILSVRC2012';
load([datasetPath '/imageListVal.mat']);
load('sizeImg_ILSVRC2014.mat');
% datasetName = 'ILSVRCtestSet';
% datasetPath = '/data/vision/torralba/deeplearning/imagenet_toolkit';
% load([datasetPath '/imageListTest.mat']);



saveFolder = ['heatMap-' datasetName '-' netName];
if ~exist(saveFolder)
    mkdir(saveFolder);
end
for i=1:5
    if ~exist([saveFolder '/top' num2str(i)])
        mkdir([saveFolder '/top' num2str(i)]);
    end
end

for i = 1:size(imageList,1)
    curImgIDX = i;
    [a b c] = fileparts(imageList{curImgIDX,1});    
    saveMatFile = [saveFolder '/' b '.mat'];
    if ~exist(saveMatFile)
        height_original = sizeFull_imageList(curImgIDX,1);%tmp.Height;
        weight_original = sizeFull_imageList(curImgIDX,2);%tmp.Width;


        curImg = imread(imageList{curImgIDX,1});

        if size(curImg,3)==1
            curImg = repmat(curImg,[1 1 3]);
        end


        scores = caffe('forward', {prepare_img(curImg, IMAGE_MEAN, CROPPED_DIM)});
        response = caffe('get_all_layers');
        scoresMean = mean(squeeze(scores{1}),2);
        [value_category, IDX_category] = sort(scoresMean,'descend');


        featureObjectSwitchSpatial = squeeze(response{end-3,1});
        [curColumnMap] = returnColumnMap(featureObjectSwitchSpatial, weights_LR(:,IDX_category(1:5)));



        for j=1:5
            curFeatureMap = squeeze(curColumnMap(:,:,j,:));
            curFeatureMap_crop = imresize(curFeatureMap,[netInfo{1,3}(1) netInfo{1,3}(2)]);
            gradients = zeros([netInfo{1,3}(1) netInfo{1,3}(2) 3 10]);
            gradients(:,:,1,:) = curFeatureMap_crop;
            gradients(:,:,2,:) = curFeatureMap_crop;
            gradients(:,:,3,:) = curFeatureMap_crop;

            [alignImgMean alignImgSet] = crop2img(gradients);
            alignImgMean = single(alignImgMean);
            alignImgMean = imresize(alignImgMean, [height_original weight_original]);
            alignImgMean = alignImgMean./max(alignImgMean(:));


            imwrite(alignImgMean, [saveFolder '/top' num2str(j) '/' b '.jpg']);

        end
        value_category = single(value_category);
        IDX_category = single(IDX_category);
        save(saveMatFile,'value_category','IDX_category');
        disp([netName ' processing ' b]);
    end
end



