Fs = 8000;
beta = 0.9;
Fc = 3600;

WavScp = args{1};
MfcScp = args{2};
WavList = textread(WavScp,'%s');
MfcList = textread(MfcScp,'%s');


WavScp_test = args{3};
MfcScp_test = args{4};
WavList_test = textread(WavScp_test,'%s');
MfcList_test = textread(MfcScp_test,'%s');

% WavList = {'/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/024o030n.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/403o031a.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/40ic020d.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/20gc010e.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/01lc020p.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/01zo031c.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/01wc0203.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_train/029o0313.wv1'};
% MfcList = {'1.MFC', '2.MFC', '3.MFC', '4.MFC', '5.MFC', '1.MFC', '2.MFC', '3.MFC', '4.MFC', '5.MFC'};
% WavList_test = {'/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_test_babble/447c020i.wv1',...
%     '/home/spapl/Database/anirudh/EE214B_project3/databases/aurora4/male_test_babble/446c0203.wv1'};
% MfcList_test = {'1.MFC', '2.MFC'};



L_train = length(WavList);
L_test = length(WavList_test);

num_frames_train = zeros(L_train,1);
num_frames_test = zeros(L_test,1);

feat_cat = [];    % Change 311 if size of Gabor feature vector changes
				   % Row size = 8000*500=4000000
% addpath('FastICA_25/');

end_frame = 0;

for k = 1:L_train
    
    srcfile = WavList{k};
%     tgtfile = MfcList{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = single(rawdata/(1.25*max(rawdata)));

    gabor = gbfb_fe(rawdata, Fs);
    
    num_frames_train(k) = size(gabor,1);
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames_train(k) - 1;
    feat_cat(start_frame:end_frame, :) = gabor;
    
%     writehtk(tgtfile,gabor,0.01,9);
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


num_feat_train = size (feat_cat,1);
end_frame = num_feat_train;
%% includes test part also for manifold mapping

for k = 1:L_test
    
    srcfile = WavList_test{k};
%     tgtfile = MfcList_test{k};
    
    fid2 = fopen(srcfile,'r');
    rawdata = fread(fid2,'int16','b');
    fclose(fid2);
    
    rawdata = rawdata/(1.25*max(rawdata));

    gabor = gbfb_fe(rawdata, Fs);
    
    num_frames_test(k) = size(gabor,1);
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames_test(k) - 1;
    feat_cat(start_frame:end_frame, :) = gabor;
    
%     writehtk(tgtfile,gabor,0.01,9);
    fprintf('file %s done.\n',srcfile);

    if(k == round(0.25*length(WavList_test)))
        disp('features extracted for 25% of testing data');
    elseif(k == round(0.5*length(WavList_test)))
        disp('features extracted for 50% of testing data');
    elseif(k == round(0.75*length(WavList_test)))
        disp('features extracted for 75% of testing data');
    elseif(k == length(WavList_test))
        disp('features extracted for 100% of testing data');
    end
    
end

outputsize = 39;

disp('Performing Laplacian EigenMap reduction on Gabor features to reduce to 39 dimension vector');

%[Coeff, score, EigenVal] = princomp(feat_cat);
%features = score(:,1:outputsize);

num_neigh = 10;

[features, ~] = leigs(feat_cat, 'nn', num_neigh, outputsize);
disp('Laplace EigenMap Dimension Reduction done...');
clear feat_cat;
disp('Writing feature files');

end_frame = 0;

for k = 1:L_train
    % srcfile = WavList{k};
    tgtfile = MfcList{k};
    
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames_train(k) - 1;
    writehtk(tgtfile,features(start_frame:end_frame,:), 0.01,9);
end

end_frame = num_feat_train;

for k = 1:L_test
    % srcfile = WavList{k};
    tgtfile = MfcList_test{k};
    
    start_frame = 1 + end_frame;
    end_frame = start_frame + num_frames_test(k) - 1;
    writehtk(tgtfile,features(start_frame:end_frame,:), 0.01,9);
end

clear features num_frames_train num_frames_test;

% fclose(fid);
