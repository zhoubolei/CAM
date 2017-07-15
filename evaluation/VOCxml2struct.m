% This code was originally written and distributed as part of the
% PASCAL VOC challenge
function res = VOCxml2struct(xml)

xml(xml==9|xml==10|xml==13)=[];

[res,xml]=parse(xml,1,[]);

function [res,ind]=parse(xml,ind,parent)

res=[];
if ~isempty(parent)&&xml(ind)~='<'
    i=findchar(xml,ind,'<');
    res=trim(xml(ind:i-1));
    ind=i;
    [tag,ind]=gettag(xml,i);
    if ~strcmp(tag,['/' parent])
        error('<%s> closed with <%s>',parent,tag);
    end
else
    while ind<=length(xml)
        [tag,ind]=gettag(xml,ind);
        if strcmp(tag,['/' parent])
            return
        else
            [sub,ind]=parse(xml,ind,tag);            
            if isstruct(sub)
                if isfield(res,tag)
                    n=length(res.(tag));
                    fn=fieldnames(sub);
                    for f=1:length(fn)
                        res.(tag)(n+1).(fn{f})=sub.(fn{f});
                    end
                else
                    res.(tag)=sub;
                end
            else
                if isfield(res,tag)
                    if ~iscell(res.(tag))
                        res.(tag)={res.(tag)};
                    end
                    res.(tag){end+1}=sub;
                else
                    res.(tag)=sub;
                end
            end
        end
    end
end

function i = findchar(str,ind,chr)

i=[];
while ind<=length(str)
    if str(ind)==chr
        i=ind;
        break
    else
        ind=ind+1;
    end
end

function [tag,ind]=gettag(xml,ind)

if ind>length(xml)
    tag=[];
elseif xml(ind)=='<'
    i=findchar(xml,ind,'>');
    if isempty(i)
        error('incomplete tag');
    end
    tag=xml(ind+1:i-1);
    ind=i+1;
else
    error('expected tag');
end 

function s = trim(s)

for i=1:numel(s)
    if ~isspace(s(i))
        s=s(i:end);
        break
    end
end
for i=numel(s):-1:1
    if ~isspace(s(i))
        s=s(1:i);
        break
    end
end

