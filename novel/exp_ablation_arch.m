% Architecture-ablation TRAINER. Trains the two ablation variants used in Table IV:
%   netC  concatenation fusion   (buildFusionNet)
%   netR  plain residual fusion  (buildGatedFusionNet, useGate=false)
% with the SAME impairment-aware augmentation/seed as run_gated, and saves them to
% novel/ablation_arch_nets.mat. The RSRP-only (netB) and gated (netF) models come
% from run_gated (novel/gated_nets.mat).
%
% NOTE: the architecture-ablation TABLE (Table IV) and the blockage TABLE (Table III)
% are both produced by run_gated.m, which scores ALL four models on a SINGLE shared
% set of 10 blockage draws so the tables agree exactly on their shared cells. This
% script only (re)generates the ablation networks.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; nP=size(D.posTrain,2); fv=D.floorVal;

augOne = @(x) applyImpairment(x, 2*rand, randi([0 8]), 0, fv);
augF = @(c) {augOne(c{1}), c{2}, c{3}};
topt = trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
    ExecutionEnvironment="cpu", Plots="none", Verbose=false);
dsF = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.posTrain),arrayDatastore(D.TtrainNN)), augF);

fprintf('Training concatenation fusion ...\n');  rng(1);
netC = trainnet(dsF, buildFusionNet(D.numSampled,nP,NB), "mse", topt);
fprintf('Training plain residual fusion (no gate) ...\n'); rng(1);
netR = trainnet(dsF, buildGatedFusionNet(D.numSampled,nP,NB,useGate=false), "mse", topt);
save(fullfile(nov,'ablation_arch_nets.mat'),'netC','netR');
fprintf('Saved netC, netR -> %s\n', fullfile(nov,'ablation_arch_nets.mat'));
fprintf('Run run_gated.m to (re)generate Tables III and IV from shared draws.\n');
disp('EXP_ABL_ARCH_TRAIN_DONE');
