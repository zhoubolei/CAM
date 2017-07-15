load('cache_groundtruth.mat');
load('/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2014_devkit/data/meta_clsloc.mat');
datasetName = 'ValSet';
datasetPath = '/data/vision/torralba/gigaSUN/deeplearning/dataset/ILSVRC2012';
load([datasetPath '/imageListVal.mat']);

ground_truth_file='../data/ILSVRC2014_clsloc_validation_ground_truth.txt';
gt_labels = dlmread(ground_truth_file);

categories = [];
for i=1:numel(synsets)
    categories{synsets(i).ILSVRC2014_ID,1} = synsets(i).words;
    categories{synsets(i).ILSVRC2014_ID,2} = synsets(i).WNID;
end

% figure
% for i=1:10
%     %curClassIDX = find(cell2mat(imageList(:,2))==i);
%     curClassIDX = find(gt_labels==i);
%     for j=1:10
%         imshow(imageList{curClassIDX(j),1});
%         waitforbuttonpress
%     end
% end
nImgs = 50000;
for i=1:nImgs
    [a b c] = fileparts(imageList{i,1});
    curPath = ['/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2012_img_val/' b c];
    curImg = imread(curPath);
    curObjects = rec(i);
    imshow(curImg);
    for j=1:numel(curObjects.objects)
        curObjectID = curObjects.objects(1,j).label;
        bbox = curObjects.objects(1,j).bbox;
        rectangle('Position',[bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)]);
        text(bbox(1), bbox(2),categories{curObjectID},'BackgroundColor',[.7 .9 .7]) 
        
    end
    waitforbuttonpress
end