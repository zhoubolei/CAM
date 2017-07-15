% this code is inspired by VOCevaldet in the PASVAL VOC devkit
% Note: this function has been significantly optimized since ILSVRC2013
function [ap recall precision] = eval_detection(predict_file,gtruth_dir,meta_file,...
                                                eval_file,blacklist_file,optional_cache_file)
% Evaluate detection
% - predict_file: each line is a single predicted object in the
%   format
%    <image_id> <ILSVRC2014_DET_ID> <confidence> <xmin> <ymin> <xmax> <ymax>
% - gtruth_dir: a path to the directory of ground truth information,
%   e.g., ILSVRC2014_DET_bbox_val/
% - meta_file: information about the synsets
% - eval_file: list of images to evaluate on
% - blacklist_file: list of image/category pairs which aren't
%    considered in evaluation
% - optional_cache_file: to save the ground truth data and avoid
%    loading from scratch again

if nargin < 3
    meta_file = '../data/meta_det.mat';
end
if nargin < 4
    eval_file = '../data/det_lists/val.txt';
end
if nargin < 5
    blacklist_file = '../data/ILSVRC2014_det_validation_blacklist.txt';
end
if nargin < 6
    optional_cache_file = ''; 
end

defaultIOUthr = 0.5;
pixelTolerance = 10;

load(meta_file);
hash = make_hash(synsets);

bLoadXML = true;
if ~isempty(optional_cache_file) && exist(optional_cache_file,'file')
    fprintf('eval_detection :: loading cached ground truth\n');
    t = tic;
    load(optional_cache_file);
    fprintf('eval_detection :: loading cached ground truth took %0.1f seconds\n',toc(t));
    if exist('gt_obj_img_ids','var')
        fprintf(['eval_detection :: loaded cache from 2014, ' ...
                 'recomputing\n']);
    else
        bLoadXML = false;
    end
end
if bLoadXML
    fprintf('eval_detection :: loading ground truth\n');
    t = tic;

    [img_basenames gt_img_ids] = textread(eval_file,'%s %d');

    num_imgs = length(img_basenames);
    gt_obj_labels = cell(1,num_imgs);
    gt_obj_bboxes = cell(1,num_imgs);
    gt_obj_thr = cell(1,num_imgs);
    num_pos_per_class = [];
    tic
    for i=1:num_imgs
        if toc > 60
            fprintf('              :: on %i of %i\n',...
                    i,num_imgs);
            tic;
        end
        rec = VOCreadxml(sprintf('%s/%s.xml',gtruth_dir, ...
                                 img_basenames{i}));
        if ~isfield(rec.annotation,'object')
            continue;
        end
        for j=1:length(rec.annotation.object)
            obj = rec.annotation.object(j);
            c = get_class2node(hash, obj.name);
            gt_obj_labels{i}(j) = c;
            if length(num_pos_per_class) < c
                num_pos_per_class(c) = 1;
            else
                num_pos_per_class(c) = num_pos_per_class(c) + 1;
            end
            b = obj.bndbox;
            bb = str2double({b.xmin b.ymin b.xmax b.ymax});
            gt_obj_bboxes{i}(:,j) = bb;
        end        

        bb = gt_obj_bboxes{i};
        gt_w = bb(4,:)-bb(2,:)+1;
        gt_h = bb(3,:)-bb(1,:)+1;
        thr = (gt_w.*gt_h)./((gt_w+pixelTolerance).*(gt_h+pixelTolerance));
        gt_obj_thr{i} = min(defaultIOUthr,thr);
    end
    fprintf('eval_detection :: loading ground truth took %0.1f seconds\n',toc(t));

    if ~isempty(optional_cache_file)
        fprintf('eval_detection :: saving cache in %s\n',optional_cache_file);
        save(optional_cache_file,'gt_img_ids','gt_obj_labels',...
             'gt_obj_bboxes','gt_obj_thr','num_pos_per_class');
    end
end

blacklist_img_id = [];
blacklist_label = [];
if ~isempty(blacklist_file) && exist(blacklist_file,'file')
    [blacklist_img_id wnid] = textread(blacklist_file,'%d %s');
    blacklist_label = zeros(length(wnid),1);
    for i=1:length(wnid)
        blacklist_label(i) = get_class2node(hash,wnid{i});        
    end
    fprintf('eval_detection :: blacklisted %i image/object pairs\n',length(blacklist_label));
else
    fprintf('eval_detection :: no blacklist\n');
end    

fprintf('eval_detection :: loading predictions\n');
t = tic;
[img_ids obj_labels obj_confs xmin ymin xmax ymax] = ...
        textread(predict_file,'%d %d %f %f %f %f %f');
obj_bboxes = [xmin ymin xmax ymax]';
fprintf('eval_detection :: loading predictions took %0.1f seconds\n',toc(t));

fprintf('eval_detection :: sorting predictions\n');
t = tic;
[img_ids ind] = sort(img_ids);
obj_confs = obj_confs(ind);
obj_labels = obj_labels(ind);
obj_bboxes = obj_bboxes(:,ind);

num_imgs = max(max(gt_img_ids),max(img_ids));
obj_labels_cell = cell(1,num_imgs);
obj_confs_cell = cell(1,num_imgs);
obj_bboxes_cell = cell(1,num_imgs);
start_i = 1;
id = img_ids(1);
tic
for i=1:length(img_ids)
    if toc > 60
        fprintf('               :: on %0.2fM of %0.2fM\n',...
                i/10^6,length(img_ids)/10^6);
        tic
    end
    if (i == length(img_ids)) || (img_ids(i+1) ~= id)
        % i is the last element of this group
        obj_labels_cell{id} = obj_labels(start_i:i)';
        obj_confs_cell{id} = obj_confs(start_i:i)';
        obj_bboxes_cell{id} = obj_bboxes(:,start_i:i);
        if i < length(img_ids)
            % start next group
            id = img_ids(i+1);
            start_i = i+1;
        end
    end
end

for i=1:num_imgs
    [obj_confs_cell{i} ind] = sort(obj_confs_cell{i},'descend');
    obj_labels_cell{i} = obj_labels_cell{i}(ind);
    obj_bboxes_cell{i} = obj_bboxes_cell{i}(:,ind);
end
tp_cell = cell(1,num_imgs);
fp_cell = cell(1,num_imgs);

fprintf('eval_detection :: sorting predictions took %0.1f seconds\n', ...
        toc(t));

fprintf('eval_detection :: accumulating\n');

num_classes = length(num_pos_per_class);

t = tic;
tic;
% iterate over images
for i=1:length(gt_img_ids)
    if toc > 60
        fprintf('               :: on %i of %i\n',...
                i,length(gt_img_ids));
        tic;
    end

    id = gt_img_ids(i);
    gt_labels = gt_obj_labels{i};
    gt_bboxes = gt_obj_bboxes{i};
    gt_thr = gt_obj_thr{i};
    num_gt_obj = length(gt_labels);
    gt_detected = zeros(1,num_gt_obj);
   
    bSameImg = blacklist_img_id == id;
    blacklisted_obj = blacklist_label(bSameImg);

    labels = obj_labels_cell{id};
    bboxes = obj_bboxes_cell{id};

    num_obj = length(labels);
    tp = zeros(1,num_obj);
    fp = zeros(1,num_obj);
    for j=1:num_obj
        if any(labels(j) == blacklisted_obj)
            continue; % just ignore this detection
        end
        bb = bboxes(:,j);
        ovmax = -inf;
        kmax = -1;
        for k=1:num_gt_obj
            if labels(j) ~= gt_labels(k)
                continue;
            end
            if gt_detected(k) > 0
                continue;
            end
            bbgt = gt_bboxes(:,k);
            bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 & ih>0                
                % compute overlap as area of intersection / area of union
                ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                   (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                   iw*ih;
                ov=iw*ih/ua;
                % makes sure that this object is detected according
                % to its individual threshold
                if ov >= gt_thr(k) && ov > ovmax
                    ovmax=ov;
                    kmax=k;
                end
            end
        end
        if kmax > 0
            tp(j) = 1;
            gt_detected(kmax) = 1;
        else
            fp(j) = 1;
        end
    end

    % put back into global vector
    tp_cell{id} = tp;
    fp_cell{id} = fp;

    for k=1:num_gt_obj
        label = gt_labels(k);
        % remove blacklisted objects from consideration as positive examples
        if any(label == blacklisted_obj) 
            num_pos_per_class(label) = num_pos_per_class(label)-1;
        end
    end
end
fprintf('eval_detection :: accumulating took %0.1f seconds\n', ...
        toc(t));

fprintf('eval_detection :: computing ap\n');
t = tic;
tp_all = [tp_cell{:}];
fp_all = [fp_cell{:}];
obj_labels = [obj_labels_cell{:}];
confs = [obj_confs_cell{:}];

[confs ind] = sort(confs,'descend');
tp_all = tp_all(ind);
fp_all = fp_all(ind);
obj_labels = obj_labels(ind);

for c=1:num_classes
    % compute precision/recall
    tp = cumsum(tp_all(obj_labels==c));
    fp = cumsum(fp_all(obj_labels==c));
    recall{c}=(tp/num_pos_per_class(c))';
    precision{c}=(tp./(fp+tp))';
    ap(c) =VOCap(recall{c},precision{c});
end

fprintf('eval_detection :: computing ap took %0.1f seconds\n', ...
        toc(t));
