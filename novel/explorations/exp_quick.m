base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);

steps = [5 10 18];   % -> 14, 7, 4 sampled beams
for st = steps
    cfg = struct('sampleStep', st);
    D = loadBeamData(fullfile(base,'data'), cfg);
    nSamp = D.numSampled;

    % ---- RSRP-only baseline ----
    rng(1);
    layers = buildRegressionNet(nSamp, D.NumBeamPairs, 96);
    opt = trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
        LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
        ValidationData={D.XvalNN,D.TvalNN}, ValidationFrequency=50, OutputNetwork="best-validation-loss", ...
        ExecutionEnvironment="cpu", Plots="none", Verbose=false);
    netB = trainnet(D.XtrainNN, D.TtrainNN, layers, "mse", opt);
    RB = evalTopK(predict(netB, D.XtestNN), D);

    % ---- fusion (RSRP + position) ----
    rng(1);
    netF = buildFusionNet(nSamp, size(D.posTrain,2), D.NumBeamPairs);
    dsTr = combine(arrayDatastore(D.XtrainNN), arrayDatastore(D.posTrain), arrayDatastore(D.TtrainNN));
    dsVa = combine(arrayDatastore(D.XvalNN),   arrayDatastore(D.posVal),   arrayDatastore(D.TvalNN));
    optF = trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
        LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
        ValidationData=dsVa, ValidationFrequency=50, OutputNetwork="best-validation-loss", ...
        ExecutionEnvironment="cpu", Plots="none", Verbose=false);
    netF = trainnet(dsTr, netF, "mse", optF);
    RF = evalTopK(predict(netF, D.XtestNN, D.posTest), D);

    fprintf('\n### sampleStep=%d  (numSampled=%d) ###\n', st, nSamp);
    fprintf('K@90:   baseline=%s  fusion=%s\n', num2str(RB.summary.K90_Neural), num2str(RF.summary.K90_Neural));
    for k = [3 5 8 13 20]
        fprintf('acc@K=%2d: baseline=%5.1f  fusion=%5.1f  (gain %+.1f)\n', ...
            k, RB.accNeural(k), RF.accNeural(k), RF.accNeural(k)-RB.accNeural(k));
    end
    fprintf('RSRP@K=8: baseline=%.2f fusion=%.2f  | KNN-only acc@K=13=%.1f\n', ...
        RB.rsrpNeural(8), RF.rsrpNeural(8), RB.accKNN(13));
end
disp('EXP_QUICK_DONE');
