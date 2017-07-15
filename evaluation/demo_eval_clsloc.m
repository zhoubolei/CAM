%this script demos the usage of evaluation routines
% the result file 'demo.val.pred.txt' on validation data is evaluated
% against the ground truth

fprintf('CLASSIFICATION WITH LOCALIZATION TASK\n');

meta_file = '../data/meta_clsloc.mat';
%pred_file='demo.val.pred.loc.txt';
%pred_file='/data/vision/torralba/deeplearning/visualization/prediction_imagenetValSet_imagenetCNNaveSumDeep.txt'
pred_file='/data/vision/torralba/deeplearning/visualization/prediction_imagenetValSet_googlenet_imagenet.txt'

ground_truth_file='../data/ILSVRC2014_clsloc_validation_ground_truth.txt';
blacklist_file='../data/ILSVRC2014_clsloc_validation_blacklist.txt';
num_predictions_per_image=5;
optional_cache_file = 'cache_groundtruth.mat';

fprintf('pred_file: %s\n', pred_file);
fprintf('ground_truth_file: %s\n', ground_truth_file);
fprintf('blacklist_file: %s\n', blacklist_file);

if isempty(optional_cache_file)
    fprintf(['NOTE: you can specify a cache filename and the ground ' ...
             'truth data will be automatically cached to save loading time ' ...
             'in the future\n']);
end

% num_val_files = -1;
% while num_val_files ~= 50000
%     if num_val_files ~= -1 
%         fprintf('That does not seem to be the correct directory. Please try again\n');
%     end
%     %ground_truth_dir=input('Please enter the path to the Validation bounding box annotations directory: ', 's');
     ground_truth_dir='/data/vision/torralba/deeplearning/imagenet_toolkit/val_bbox';
%     %fprintf('ground_truth_dir: %s\n', ground_truth_dir);
%     val_files = dir(sprintf('%s/*.xml',ground_truth_dir));
%     num_val_files = numel(val_files);
% end

error_cls = zeros(num_predictions_per_image,1);
error_loc = zeros(num_predictions_per_image,1);

for i=1:num_predictions_per_image
    [error_cls(i) error_loc(i)] = eval_clsloc(pred_file,ground_truth_file,ground_truth_dir,...
                                              meta_file,i, blacklist_file,optional_cache_file);
end

disp('# guesses vs clsloc error vs cls-only error');
disp([(1:num_predictions_per_image)',error_loc,error_cls]);

%% Result: alexFullConv network
% # guesses vs clsloc error vs cls-only error
%     1.0000    0.7461    0.5077
%     2.0000    0.6859    0.3841
%     3.0000    0.6569    0.3209
%     4.0000    0.6401    0.2835
%     5.0000    0.6270    0.2556

%% Result: googleNet_imagenet
% # guesses vs clsloc error vs cls-only error
%     1.0000    0.6691    0.3450
%     2.0000    0.6158    0.2294
%     3.0000    0.5937    0.1786
%     4.0000    0.5813    0.1503
%     5.0000    0.5734    0.1314