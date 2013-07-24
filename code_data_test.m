Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};
WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');

for k = 1:length(WavList)
    srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = rawdata/(1.25*max(rawdata));
    
%    mfccs = PNCC(rawdata,Fs);       % Modify to change function call (same as code_data_train.m)
 [mfccs, ~]= gbfb_fe(rawdata,Fs); 
    writehtk(tgtfile,mfccs,0.01,9);
    fprintf('file %s done.\n',srcfile);

    if(k == round(0.25*length(WavList)))
        disp('features extracted for 25% of testing data');
    elseif(k == round(0.5*length(WavList)))
        disp('features extracted for 50% of testing data');
    elseif(k == round(0.75*length(WavList)))
        disp('features extracted for 75% of testing data');
    elseif(k == length(WavList))
        disp('features extracted for 100% of testing data');
    end
end
