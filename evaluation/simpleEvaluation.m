function [error_clsSet, error_locSet] = simpleEvaluation(pred_input)

max_num_pred_per_imageBound = 5;
gtruth_dir='/data/vision/torralba/deeplearning/imagenet_toolkit/val_bbox';
meta_file = '/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2014_devkit/data/meta_clsloc.mat';
gtruth_file='/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt';
blacklist_file='/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_blacklist.txt';
optional_cache_file = '/data/vision/torralba/deeplearning/imagenet_toolkit/ILSVRC2014_devkit/evaluation/cache_groundtruth.mat';


bEvalLoc = nargout > 1;

bLoadXML = bEvalLoc;
if bEvalLoc && ~isempty(optional_cache_file) && exist(optional_cache_file,'file')
 %   fprintf('eval_clsloc :: loading cached ground truth\n');
    t = tic;
    load(optional_cache_file);
%    fprintf('eval_clsloc :: loading cached ground truth took %0.1f seconds\n',toc(t));
    bLoadXML = false;
end
if bLoadXML
 %   fprintf('eval_clsloc :: loading ground truth\n');
  %  t = tic;

    load (meta_file);
    hash = make_hash(synsets);

 %   tic
    gt = dir(sprintf('%s/*.xml',gtruth_dir));
    for i=1:length(gt)
 %       if toc > 60
 %           fprintf('              :: on %i of %i\n',...
 %                   i,length(gt));
 %           tic;
 %       end
        filename = gt(i).name;
        r = VOCreadrecxml(sprintf('%s/%s',gtruth_dir,filename),hash);
        objs = rmfield(r.objects,{'class','bndbox'});
        rec(i).objects = objs;
    end
%    fprintf(['eval_clsloc :: loading ground truth took %0.1f seconds\n'],toc(t));
    if ~isempty(optional_cache_file)
 %       fprintf('eval_clsloc :: saving cache in %s\n',optional_cache_file);
        save(optional_cache_file,'rec');
    end
end

%pred = dlmread(predict_file);

% for i=1:num_predictions_per_image
%     [error_cls(i) error_loc(i)] = eval_clsloc(pred_file,ground_truth_file,ground_truth_dir,...
%                                               meta_file,i, blacklist_file,optional_cache_file);
% end
% 
% disp('# guesses vs clsloc error vs cls-only error');
% disp([(1:num_predictions_per_image)',error_loc,error_cls]);

error_clsSet = zeros(max_num_pred_per_imageBound,1);
error_locSet = zeros(max_num_pred_per_imageBound,1);

for max_num_pred_per_image = 1:max_num_pred_per_imageBound
    
    pred = pred_input;
    gt_labels = dlmread(gtruth_file);
    n = size(gt_labels,2);
    %% extra labels are ignored
    if size(pred_input,2) > max_num_pred_per_image*5
        pred = pred_input(:,1:max_num_pred_per_image*5);
    end
    assert(size(pred,1)==size(gt_labels,1));
    num_guesses = size(pred,2)/5;

    pred_labels = pred(:,1:5:end);

    %pred_bbox = zeros(size(pred,1), num_guesses, 4);

    %for i=1:5:size(pred,2)
    %    pred_labels = [ pred_labels, pred(:,i) ];
    %    for j=1:size(pred,1)
    %        pred_bbox(j,ceil(i/5),:) = pred(j,i+1:i+4);
    %    end
    %end

    % compute classification error
    c = zeros(size(pred_labels,1),1);
    for j=1:size(gt_labels,2) %for each ground truth label
        x = gt_labels(:,j) * ones(1,size(pred_labels,2));
        c = c + min( x ~= pred_labels, [], 2);
    end
    n = sum(gt_labels~=0,2);
    cls_error = sum(c./n)/size(pred_labels,1);

    error_clsSet(max_num_pred_per_image) = cls_error;
    
    if ~bEvalLoc
        loc_error = 0;
        return;
    end

    % compute localization error

    blacklist = [];
    blacklist_size = 0;
    if exist('blacklist_file','var') && exist(blacklist_file,'file')
        blacklist = dlmread(blacklist_file);
        blacklist_size = length(blacklist);
    %    fprintf('eval_clsloc :: blacklisted %i images\n',blacklist_size);
    else
    %    fprintf('eval_clsloc :: no blacklist\n');
    end
    blacklist_mask = zeros(size(gt_labels,1), 1);
    blacklist_mask(blacklist) = 1;

    t = tic;
    e = zeros(size(pred_labels,1),1);

    for i=1:size(pred,1)				
        if toc(t) > 60
    %        fprintf('  eval_clsloc :: on %i of %i\n',i, ...
    %                size(pred,1));
            t = tic;
        end
        e(i) = 0;
        if blacklist_mask(i) > 0
            continue
        end
        for k=1:n % sum
            for j=1:num_guesses % min
                d_jk = (gt_labels(i,k) ~= pred_labels(i,j));
                if d_jk == 0
                    box = pred(i,(j-1)*5+1+(1:4)); % j^th predicted box
                    ov_vector = compute_overlap(box,rec(i),gt_labels(i,k));	
                    f_j = ( ov_vector < 0.50 );
                else
                    f_j = 1;
                end
                d_jk = ones(1,numel(f_j)) * d_jk;
                d(i,j) = min( max([f_j;d_jk]) );
            end		
            e(i) = e(i) + min(d(i,:));	%% min over j
        end
    end
    clsloc_error = sum(e./n)/(size(pred_labels,1) - blacklist_size);
    error_locSet(max_num_pred_per_image) = clsloc_error;
end
