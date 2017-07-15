% This code was originally written and distributed as part of the
% PASCAL VOC challenge, with minor modifications for ILSVRC2013
function rec = VOCreadrecxml(path,hash)

x=VOCreadxml(path);
x=x.annotation;

rec.folder=x.folder;
rec.filename=x.filename;
rec.source.database=x.source.database;
%% rec.source.annotation=x.source.annotation;
%% rec.source.image=x.source.image;

rec.size.width=str2double(x.size.width);
rec.size.height=str2double(x.size.height);

rec.imgname=[x.folder x.filename];
rec.imgsize=str2double({x.size.width x.size.height});
rec.database=rec.source.database;

if isfield(x,'object')
    for i=1:length(x.object)
        rec.objects(i)=xmlobjtopas(x.object(i),hash);
    end
else
    rec.objects = [];
end

function p = xmlobjtopas(o,hash)

p.class=o.name;

p.label= get_class2node( hash, p.class );

p.bbox=str2double({o.bndbox.xmin o.bndbox.ymin o.bndbox.xmax o.bndbox.ymax});

p.bndbox.xmin=str2double(o.bndbox.xmin);
p.bndbox.ymin=str2double(o.bndbox.ymin);
p.bndbox.xmax=str2double(o.bndbox.xmax);
p.bndbox.ymax=str2double(o.bndbox.ymax);

