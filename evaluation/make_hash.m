function dict = make_hash(synsets)

dict = java.util.Hashtable;
for i = 1:numel(synsets)
    dict.put(synsets(i).WNID, i);
end

