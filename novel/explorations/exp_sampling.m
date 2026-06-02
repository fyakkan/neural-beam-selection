base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);

% --- compute candidate 14-beam selections from TRAINING data ---
tr = load(fullfile(base,'data','nnBS_TrainingData.mat'));
S  = load(fullfile(base,'data','nnBS_prm.mat')); NB = S.prm.NumTxBeams*S.prm.NumRxBeams;
R  = double(reshape(tr.rsrpMatTrain, NB, []));      % 70 x N
Rf = R; Rf(~isfinite(Rf)) = -120;
nSel = 14;

sel.every5   = 1:5:NB;                                            % baseline (14)
[~,o] = sort(var(Rf,0,2),'descend');         sel.hiVar   = sort(o(1:nSel))';   % most variable
cnt = accumarray(tr.optBeamPairIdxTrain(:),1,[NB 1]);
[~,o] = sort(cnt,'descend');                 sel.freqOpt = sort(o(1:nSel))';   % most-often optimal
[~,o] = sort(mean(Rf,2),'descend');          sel.hiMean  = sort(o(1:nSel))';   % strongest mean RSRP
% greedy max-coverage in 2D grid via farthest-point on (row,col) is overkill; skip.

names = fieldnames(sel);
for i = 1:numel(names)
    idx = sel.(names{i});
    D = loadBeamData(fullfile(base,'data'), struct('sampIdx', idx));
    rng(1);
    layers = buildRegressionNet(D.numSampled, D.NumBeamPairs, 96);
    opt = trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
        LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
        ValidationData={D.XvalNN,D.TvalNN}, ValidationFrequency=50, OutputNetwork="best-validation-loss", ...
        ExecutionEnvironment="cpu", Plots="none", Verbose=false);
    net = trainnet(D.XtrainNN, D.TtrainNN, layers, "mse", opt);
    RR = evalTopK(predict(net, D.XtestNN), D);
    fprintf('%-9s idx=%s\n', names{i}, mat2str(idx));
    fprintf('   K@90=%2s  acc@K=5=%.1f  @8=%.1f  @13=%.1f  @20=%.1f\n', ...
        num2str(RR.summary.K90_Neural), RR.accNeural(5), RR.accNeural(8), RR.accNeural(13), RR.accNeural(20));
end
disp('EXP_SAMPLING_DONE');
