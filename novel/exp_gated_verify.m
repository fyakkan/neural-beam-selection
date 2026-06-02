% Verify the gated fusion: clean parity with RSRP-only + win under beam blockage.
% Both models trained with the SAME blockage-focused augmentation.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; Nte=D.Ntest; trueOpt=D.optTest(:); fv=D.floorVal;

% blockage-focused augmentation: drop 0..8 of 14 input beams, light noise, no quant
augOne = @(x) applyImpairment(x, 2*rand, randi([0 8]), 0, fv);
augR = @(c) {augOne(c{1}), c{2}};
augF = @(c) {augOne(c{1}), c{2}, c{3}};
opts = {"adam","MaxEpochs",300,"MiniBatchSize",512,"InitialLearnRate",1e-3, ...
    "LearnRateSchedule","piecewise","LearnRateDropFactor",0.5,"LearnRateDropPeriod",80, ...
    "ExecutionEnvironment","cpu","Plots","none","Verbose",false};

rng(1);
dsR = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.TtrainNN)), augR);
netB = trainnet(dsR, buildRegressionNet(D.numSampled,NB,96), "mse", trainingOptions(opts{:}));
rng(1);
dsF = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.posTrain),arrayDatastore(D.TtrainNN)), augF);
netF = trainnet(dsF, buildGatedFusionNet(D.numSampled,size(D.posTrain,2),NB), "mse", trainingOptions(opts{:}));
save(fullfile(nov,'gated_verify_nets.mat'),'netB','netF');

accAtK = @(pred,k) 100*mean(arrayfun(@(n) any(mk(pred(n,:),k)==trueOpt(n)), 1:Nte));

fprintf('\n==== Gated fusion vs RSRP-only: acc@K=13, blockage sweep (sigma=0) ====\n');
fprintf('%8s | %-9s | %-9s | gain\n','#blocked','RSRP-only','GatedFus');
for nB=[0 1 2 4 6 8 10 12]
    [aB,aF]=evalSetting(netB,netF,D,0,nB,accAtK,12);
    fprintf('%8d | %8.1f  | %8.1f  | %+.1f\n', nB, aB, aF, aF-aB);
end
disp('EXP_GATED_DONE');

% ===================== local functions =====================
function idx = mk(v,k); [~,idx] = maxk(v,k); end
function [aB,aF] = evalSetting(netB,netF,D,sigma,nB,accAtK,nrep)
    aB=0; aF=0;
    for r=1:nrep
        Xn = applyImpairment(D.XtestNN, sigma, nB, 0, D.floorVal);
        aB = aB + accAtK(predict(netB,Xn),13)/nrep;
        aF = aF + accAtK(predict(netF,Xn,D.posTest),13)/nrep;
    end
end
