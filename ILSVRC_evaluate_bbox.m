

datasetName = 'ILSVRCvalSet';
load('imagenet_toolkit/ILSVRC2014_devkit/evaluation/cache_groundtruth.mat');
load('imagenet_toolkit/ILSVRC2014_devkit/data/meta_clsloc.mat');
% download the toolkit at http://www.image-net.org/challenges/LSVRC/2014/
datasetPath = 'dataset/ILSVRC2012';
load([datasetPath '/imageListVal.mat']);
load('sizeImg_ILSVRC2014.mat');

% datasetName = 'ILSVRCtestSet';
% datasetPath = '/data/vision/torralba/deeplearning/imagenet_toolkit';
% load([datasetPath '/imageListTest.mat']);


nImgs = size(imageList,1);

ground_truth_file='imagenet_toolkit/ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt';
gt_labels = dlmread(ground_truth_file);

categories_gt = [];
categoryIDMap = containers.Map();
for i=1:numel(synsets)
    categories_gt{synsets(i).ILSVRC2014_ID,1} = synsets(i).words;
    categories_gt{synsets(i).ILSVRC2014_ID,2} = synsets(i).WNID;
    categoryIDMap(synsets(i).WNID) = i;
end



%% network to evaluate
% backpropa-heatmap
%netName = 'caffeNet_imagenet';
%netName = 'googlenetBVLC_imagenet';
%netName = 'VGG16_imagenet';

% CAM-based network
%netName = 'NIN';
%netName = 'CAM_imagenetCNNaveSumDeep';
%netName = 'CAM_googlenetBVLC_imagenet';% the direct output
netName = 'CAM_googlenetBVLCshrink_imagenet';
%netName = 'CAM_googlenetBVLCshrink_imagenet_maxpool';
%netName = 'CAM_VGG16_imagenet';
%netName = 'CAM_alexnet';

load('categoriesImageNet.mat');

visualizationPointer = 0;

topCategoryNum = 5;
predictionResult_bbox1 = zeros(nImgs, topCategoryNum*5);
predictionResult_bbox2 = zeros(nImgs, topCategoryNum*5);
predictionResult_bboxCombine = zeros(nImgs, topCategoryNum*5);

if matlabpool('size')==0
    try
        matlabpool
    catch e
    end
end

heatMapFolder = ['heatMap-' datasetName '-' netName];
bbox_threshold = [20, 100, 110];
curParaThreshold = [num2str(bbox_threshold(1)) ' ' num2str(bbox_threshold(2)) ' ' num2str(bbox_threshold(3))];
parfor i=1:size(imageList,1)
    curImgIDX = i;

    height_original = sizeFull_imageList(curImgIDX,1);%tmp.Height;
    weight_original = sizeFull_imageList(curImgIDX,2);%tmp.Width;
    
    [a b c] = fileparts(imageList{curImgIDX,1});
    curPath_fullSizeImg = ['/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2012_img_val/' b c];
    curMatFile = [heatMapFolder '/' b '.mat'];
    [heatMapSet, value_category, IDX_category] = loadHeatMap( curMatFile);
    
    curResult_bbox1 = [];
    curResult_bbox2 = [];
    curResult_bboxCombine = [];
    for j=1:5
        curHeatMapFile = [heatMapFolder '/top' num2str(j) '/' b '.jpg'];

        curBBoxFile = [heatMapFolder '/top' num2str(j) '/' b '_default.txt'];
        %curBBoxFileGraphcut = [heatMapFolder '/top' num2str(j) '/' b '_graphcut.txt'];
        curCategory = categories{IDX_category(j),1};
        %imwrite(curHeatMap, ['result_bbox/heatmap_tmp' b randString '.jpg']);
        if ~exist(curBBoxFile)
            %system(['/data/vision/torralba/deeplearning/package/bbox_hui/final ' curHeatMapFile ' ' curBBoxFile]);
            
            system(['/data/vision/torralba/deeplearning/package/bbox_hui_new/./dt_box ' curHeatMapFile ' ' curParaThreshold ' ' curBBoxFile]);
        end
        curPredictCategory = categories{IDX_category(j),1};
        curPredictCategoryID = categories{IDX_category(j),1}(1:9);
        curPredictCategoryGTID = categoryIDMap(curPredictCategoryID);
        
        
        boxData = dlmread(curBBoxFile);
        boxData_formulate = [boxData(1:4:end)' boxData(2:4:end)' boxData(1:4:end)'+boxData(3:4:end)' boxData(2:4:end)'+boxData(4:4:end)'];
        boxData_formulate = [min(boxData_formulate(:,1),boxData_formulate(:,3)),min(boxData_formulate(:,2),boxData_formulate(:,4)),max(boxData_formulate(:,1),boxData_formulate(:,3)),max(boxData_formulate(:,2),boxData_formulate(:,4))];
           
%         try
%             boxDataGraphcut = dlmread(curBBoxFileGraphcut);
%             boxData_formulateGraphcut = [boxDataGraphcut(1:4:end)' boxDataGraphcut(2:4:end)' boxDataGraphcut(1:4:end)'+boxDataGraphcut(3:4:end)' boxDataGraphcut(2:4:end)'+boxDataGraphcut(4:4:end)'];
%         catch exception
%             boxDataGraphcut = dlmread(curBBoxFile);
%             boxData_formulateGraphcut = [boxDataGraphcut(1:4:end)' boxDataGraphcut(2:4:end)' boxDataGraphcut(1:4:end)'+boxDataGraphcut(3:4:end)' boxDataGraphcut(2:4:end)'+boxDataGraphcut(4:4:end)'];
%             boxData_formulateGraphcut = boxData_formulateGraphcut(1,:);
%         end

        bbox = boxData_formulate(1,:); 
        curPredictTuple = [curPredictCategoryGTID bbox(1) bbox(2) bbox(3) bbox(4)];
        curResult_bbox1 = [curResult_bbox1 curPredictTuple];
        curResult_bboxCombine = [curResult_bboxCombine curPredictTuple];
        
        bbox = boxData_formulate(2,:); 
        %bbox = boxData_formulateGraphcut(1,:);
        curPredictTuple = [curPredictCategoryGTID bbox(1) bbox(2) bbox(3) bbox(4)];
        curResult_bbox2 = [curResult_bbox2 curPredictTuple];      
        
        curResult_bboxCombine = [curResult_bboxCombine curPredictTuple];
        if visualizationPointer == 1
              
            curHeatMap = imread(curHeatMapFile);
            curHeatMap = imresize(curHeatMap,[height_original weight_original]);
        
            subplot(1,2,1),hold off, imshow(curPath_fullSizeImg);
            hold on
            curBox = boxData_formulate(1,:);
            rectangle('Position',[curBox(1) curBox(2) curBox(3)-curBox(1) curBox(4)-curBox(2)],'EdgeColor',[1 0 0]);
            subplot(1,2,2),imagesc(curHeatMap);
            title(curCategory);
            waitforbuttonpress
        end
    end
    
    predictionResult_bbox1(i, :) = curResult_bbox1;
    predictionResult_bbox2(i, :) = curResult_bbox2;
    predictionResult_bboxCombine(i,:) = curResult_bboxCombine(1:topCategoryNum*5);
    disp([netName ' processing ' b])
end


addpath('evaluation');
disp([netName '--------bbox1' ]);
[cls_error, clsloc_error] = simpleEvaluation(predictionResult_bbox1);
disp([(1:5)',clsloc_error,cls_error]);

disp([netName '--------bbox2' ]);
[cls_error, clsloc_error] = simpleEvaluation(predictionResult_bbox2);
disp([(1:5)',clsloc_error,cls_error]);

disp([netName '--------bboxCombine' ]);
[cls_error, clsloc_error] = simpleEvaluation(predictionResult_bboxCombine);
disp([(1:5)',clsloc_error,cls_error]);
