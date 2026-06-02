% Confirming experiment: does multi-modal fusion beat RSRP-only under realistic
% RSRP impairments (noise / quantization / beam blockage)? Both models are
% trained impairment-aware with the SAME augmentation; only the input modality differs.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; Nte=D.Ntest; trueOpt=D.optTest(:); fv=D.floorVal;

% ---- impairment-aware training augmentation (random levels per sample) ----
pTr = struct('sigMax',8,'bMax',8,'q',2,'fv',fv);
augOne = @(x) applyImpairment(x, pTr.sigMax*rand, randi([0 pTr.bMax]), pTr.q, pTr.fv);
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
netF = trainnet(dsF, buildFusionNet(D.numSampled,size(D.posTrain,2),NB), "mse", trainingOptions(opts{:}));
save(fullfile(nov,'impair_nets.mat'),'netB','netF','pTr');

accAtK = @(pred,k) 100*mean(arrayfun(@(n) any(mk(pred(n,:),k)==trueOpt(n)), 1:Nte));

fprintf('\n==== acc@K=13: RSRP-only vs Fusion (impairment-aware training) ====\n');
fprintf('\n[A] BEAM BLOCKAGE sweep (out of 14 inputs; sigma=2dB, q=1dB):\n');
fprintf('%8s | %-9s | %-9s | gain\n','#blocked','RSRP-only','Fusion');
for nB=[0 2 4 6 8 10]
    [aB,aF]=evalSetting(netB,netF,D,2,nB,1,accAtK,8);
    fprintf('%8d | %8.1f  | %8.1f  | %+.1f\n', nB, aB, aF, aF-aB);
end

fprintf('\n[B] NOISE sweep (sigma dB; 0 blocked, q=1dB):\n');
fprintf('%8s | %-9s | %-9s | gain\n','sigma','RSRP-only','Fusion');
for sg=[0 2 4 6 8 10]
    [aB,aF]=evalSetting(netB,netF,D,sg,0,1,accAtK,8);
    fprintf('%8.0f | %8.1f  | %8.1f  | %+.1f\n', sg, aB, aF, aF-aB);
end

fprintf('\n[C] QUANTIZATION sweep (step dB; sigma=2dB, 2 blocked):\n');
fprintf('%8s | %-9s | %-9s | gain\n','q','RSRP-only','Fusion');
for q=[0 1 2 4 6]
    [aB,aF]=evalSetting(netB,netF,D,2,2,q,accAtK,8);
    fprintf('%8.0f | %8.1f  | %8.1f  | %+.1f\n', q, aB, aF, aF-aB);
end

fprintf('\n[D] REALISTIC combined (sigma=4dB, 3 blocked, q=2dB):\n');
[aB,aF]=evalSetting(netB,netF,D,4,3,2,accAtK,20);
fprintf('   RSRP-only=%.1f  Fusion=%.1f  gain=%+.1f\n', aB, aF, aF-aB);
fprintf('\n[E] CLEAN (sigma=0,0 blocked,q=0):\n');
[aB,aF]=evalSetting(netB,netF,D,0,0,0,accAtK,1);
fprintf('   RSRP-only=%.1f  Fusion=%.1f  gain=%+.1f\n', aB, aF, aF-aB);
disp('EXP_IMPAIR_DONE');

% ===================== local functions (must be at end) =====================
function idx = mk(v,k)
    [~,idx] = maxk(v,k);
end

function [aB,aF] = evalSetting(netB,netF,D,sigma,nB,q,accAtK,nrep)
    aB=0; aF=0;
    for r=1:nrep
        Xn = applyImpairment(D.XtestNN, sigma, nB, q, D.floorVal);
        aB = aB + accAtK(predict(netB,Xn),13)/nrep;
        aF = aF + accAtK(predict(netF,Xn,D.posTest),13)/nrep;
    end
end
