Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};


WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

%WavList = {'/home/anirudh/Dropbox/acads/Speech_214B/EE214B_project/databases/aurora4/male_train/024o030n.wv1'};
%MfcList = {''};

for k = 1:length(WavList)
    
    srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = rawdata/(1.25*max(rawdata));
    
     [mfccs, ~]= gbfb_fe(rawdata,Fs);      % Modify to change function call
    
  writehtk(tgtfile,mfccs,0.01,9);
   fprintf('file %s done.\n',srcfile);
   
   if(k == round(0.25*length(WavList)))
       disp('features extracted for 25% of training data');
   elseif(k == round(0.5*length(WavList)))
       disp('features extracted for 50% of training data');
   elseif(k == round(0.75*length(WavList)))
       disp('features extracted for 75% of training data');
   elseif(k == length(WavList))
       disp('features extracted for 100% of training data');
   end
    
end

