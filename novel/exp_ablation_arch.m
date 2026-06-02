% Architecture ablation: is the GATE needed? Compare under beam blockage:
%   RSRP-only | Concatenation fusion | Plain residual fusion | Gated residual fusion
% All trained impairment-aware with the SAME augmentation/seed as run_gated.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; nP=size(D.posTrain,2); fv=D.floorVal;

% reuse already-trained RSRP-only + gated nets
L = load(fullfile(nov,'gated_nets.mat')); netB=L.netB; netG=L.netF;

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

fprintf('\n==== acc@K=13 vs beam blockage (architecture ablation) ====\n');
fprintf('%8s | %-9s | %-7s | %-8s | %-8s\n','#blocked','RSRP-only','Concat','Residual','Gated');
rng(7);
for nb=[0 2 4 6 8 10 12]
    a = zeros(1,4); nrep=10;
    for r=1:nrep
        Xn = applyImpairment(D.XtestNN, 0, nb, 0, fv);
        a(1)=a(1)+acc13(predict(netB,Xn),D)/nrep;
        a(2)=a(2)+acc13(predict(netC,Xn,D.posTest),D)/nrep;
        a(3)=a(3)+acc13(predict(netR,Xn,D.posTest),D)/nrep;
        a(4)=a(4)+acc13(predict(netG,Xn,D.posTest),D)/nrep;
    end
    fprintf('%8d | %8.1f  | %6.1f | %7.1f  | %7.1f\n', nb, a(1),a(2),a(3),a(4));
end
disp('EXP_ABL_ARCH_DONE');

function a = acc13(pred, D)
    [~,ord]=sort(pred,2,'descend'); t=D.optTest(:); h=0;
    for n=1:D.Ntest, if any(ord(n,1:13)==t(n)), h=h+1; end; end
    a=100*h/D.Ntest;
end
