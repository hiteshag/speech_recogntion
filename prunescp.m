% removing names from the script file that do not appear in the MLF

MLF = args{1};
ipScp = args{2};
opScp = args{3};

labelNames = textread(MLF,'%s');
ipScpNames = textread(ipScp,'%s');
fid = fopen(opScp,'w');

for i = 1:length(ipScpNames)
    
    for j = 1:length(labelNames)
        
        if(strcmp(ipScpNames{i}(1:end-4),labelNames{j}(2:end-5)))
            
            fprintf(fid,'%s\n',ipScpNames{i});
            break;
            
        end
        
    end
    
end

fclose(fid);


