% Architecture ablation: is the GATE needed? Compare under beam blockage:
%   RSRP-only | Concatenation fusion | Plain residual fusion | Gated residual fusion
% All trained impairment-aware with the SAME augmentation/seed as run_gated.
% Reports mean +/- std of top-K=13 accuracy over nrep random blockage draws.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; nP=size(D.posTrain,2); fv=D.floorVal;

% reuse already-trained RSRP-only + gated nets
L = load(fullfile(nov,'gated_nets.mat')); netB=L.netB; netG=L.netF;

% reuse saved concat/residual nets if present (reproduces published means); else train
arcFile = fullfile(nov,'ablation_arch_nets.mat');
if exist(arcFile,'file')
    A = load(arcFile); netC=A.netC; netR=A.netR;
    fprintf('Loaded concat/residual nets from %s\n', arcFile);
else
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
    save(arcFile,'netC','netR');
end

levels=[0 2 4 6 8 10 12]; nrep=10; nL=numel(levels);
A_all=zeros(4,nL,nrep);   % models x levels x draws
rng(7);
for i=1:nL
    nb=levels(i);
    for r=1:nrep
        Xn = applyImpairment(D.XtestNN, 0, nb, 0, fv);
        A_all(1,i,r)=acc13(predict(netB,Xn),D);
        A_all(2,i,r)=acc13(predict(netC,Xn,D.posTest),D);
        A_all(3,i,r)=acc13(predict(netR,Xn,D.posTest),D);
        A_all(4,i,r)=acc13(predict(netG,Xn,D.posTest),D);
    end
end
M=mean(A_all,3); S=std(A_all,0,3);   % 4 x nL
names={'RSRP-only','Concat','Residual','Gated'};

fprintf('\n==== acc@K=13 (mean +/- std over %d draws) vs beam blockage ====\n', nrep);
fprintf('%-10s', '#blocked'); fprintf(' | %s', sprintf('%d',levels)); fprintf('\n');
for m=1:4
    fprintf('%-12s', names{m});
    for i=1:nL, fprintf(' %5.1f+/-%3.1f', M(m,i), S(m,i)); end
    fprintf('\n');
end
fprintf('\nLaTeX Table IV rows (mean$\\pm$std) for levels [0 6 8 10 12]:\n');
sel=ismember(levels,[0 6 8 10 12]);
for m=1:4
    fprintf('  %-14s & %s \\\\\n', names{m}, latexRow(M(m,sel),S(m,sel)));
end
disp('EXP_ABL_ARCH_DONE');

function a = acc13(pred, D)
    [~,ord]=sort(pred,2,'descend'); t=D.optTest(:); h=0;
    for n=1:D.Ntest, if any(ord(n,1:13)==t(n)), h=h+1; end; end
    a=100*h/D.Ntest;
end

function s = latexRow(m, sd)
    parts = arrayfun(@(a,b) sprintf('$%.1f\\pm%.1f$', a, b), m, sd, 'uni', 0);
    s = strjoin(parts, ' & ');
end
