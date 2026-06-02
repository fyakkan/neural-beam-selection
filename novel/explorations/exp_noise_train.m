base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; Nte=D.Ntest; trueOpt=D.optTest(:);
sigMaxTrain = 10;   % train with RSRP noise sigma ~ U[0, sigMaxTrain] dB (per sample)

baseOpt = {"adam","MaxEpochs",300,"MiniBatchSize",512,"InitialLearnRate",1e-3, ...
    "LearnRateSchedule","piecewise","LearnRateDropFactor",0.5,"LearnRateDropPeriod",80, ...
    "ExecutionEnvironment","cpu","Plots","none","Verbose",false};

% noise augmentation transforms (fresh noise per read; per-sample sigma)
augR = @(c) {c{1}+ (sigMaxTrain*rand)*randn(size(c{1})), c{2}};
augF = @(c) {c{1}+ (sigMaxTrain*rand)*randn(size(c{1})), c{2}, c{3}};

% ---- RSRP-only, noise-aware ----
rng(1);
dsR = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.TtrainNN)), augR);
netB = trainnet(dsR, buildRegressionNet(D.numSampled,NB,96), "mse", trainingOptions(baseOpt{:}));

% ---- fusion, noise-aware ----
rng(1);
dsF = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.posTrain),arrayDatastore(D.TtrainNN)), augF);
netF = trainnet(dsF, buildFusionNet(D.numSampled,size(D.posTrain,2),NB), "mse", trainingOptions(baseOpt{:}));

accAtK = @(pred,k) 100*mean(arrayfun(@(n) any(mk(pred(n,:),k)==trueOpt(n)), 1:Nte));
fprintf('\nNoise-AWARE training (sigma~U[0,%g] dB). Test acc@K=13 (avg of 5 noise draws):\n', sigMaxTrain);
fprintf('%6s | %-12s | %-12s | %s\n','sigma','RSRP-only','Fusion','gain');
rng(7);
for sigma=[0 2 4 6 8 10 12]
    aB=0;aF=0; nrep=5;
    for r=1:nrep
        Xn=D.XtestNN+sigma*randn(size(D.XtestNN));
        aB=aB+accAtK(predict(netB,Xn),13)/nrep;
        aF=aF+accAtK(predict(netF,Xn,D.posTest),13)/nrep;
    end
    fprintf('%6.0f | %10.1f   | %10.1f   | %+.1f\n', sigma, aB, aF, aF-aB);
end
disp('EXP_NOISE_TRAIN_DONE');
function idx=mk(v,k); [~,idx]=maxk(v,k); end
