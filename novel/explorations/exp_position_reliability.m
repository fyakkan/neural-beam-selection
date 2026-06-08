% Does the gate do ADAPTIVE work when position reliability VARIES per UE?
% Regime: fixed beam blockage (nB=8, so position is useful) PLUS per-UE position
% outage (with prob p the UE has no GPS fix -> position set to an out-of-range
% sentinel). Both gated and plain-residual fusion are trained with this combined
% augmentation. We test whether (a) the learned gate CLOSES for outaged UEs and
% OPENS for reliable ones (per-UE adaptivity), and (b) the gated model retains
% accuracy that the fixed-scale plain residual loses as the outage rate grows.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; fv=D.floorVal; nP=size(D.posTrain,2);

% clean position normalisation (manual; outage uses an out-of-range sentinel)
mu = mean(D.posTrain,1); sd = std(D.posTrain,0,1);
normPos = @(P) (P - mu)./sd;
posTrN = normPos(D.posTrain); posTeN = normPos(D.posTest);
sentinel = -8*ones(1,nP);            % far outside the ~[-3,3] z-range => detectable outage
pCorruptTrain = 0.4; nBfix = 8;

% RSRP-only floor: reuse the blockage-trained baseline (position-agnostic)
L = load(fullfile(nov,'gated_nets.mat')); netB = L.netB;

% training augmentation: RSRP blockage + per-UE position outage
augOne = @(x) applyImpairment(x, 2*rand, randi([0 8]), 0, fv);
outRow = @(p) ternary(rand < pCorruptTrain, sentinel, p);
augG = @(c) {augOne(c{1}), outRow(c{2}), c{3}};
topt = trainingOptions("adam", MaxEpochs=300, MiniBatchSize=512, InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, LearnRateDropPeriod=80, ...
    ExecutionEnvironment="cpu", Plots="none", Verbose=false);
dsTr = combine(arrayDatastore(D.XtrainNN), arrayDatastore(posTrN), arrayDatastore(D.TtrainNN));

fprintf('Training GATED fusion (blockage + position-outage aug) ...\n'); rng(1);
netG = trainnet(transform(dsTr,augG), buildGatedFusionNet(D.numSampled,nP,NB,posNorm='none'), "mse", topt);
fprintf('Training PLAIN-RESIDUAL fusion (no gate, same aug) ...\n'); rng(1);
netR = trainnet(transform(dsTr,augG), buildGatedFusionNet(D.numSampled,nP,NB,useGate=false,posNorm='none'), "mse", topt);
save(fullfile(nov,'posrel_nets.mat'),'netG','netR','mu','sd','sentinel');

% ---- evaluation: fixed blockage nB=8, sweep position-outage rate ----
pgrid=[0 0.25 0.5 0.75 1.0]; nrep=10; nL=numel(pgrid);
aG=zeros(nL,nrep); aR=zeros(nL,nrep); aB=zeros(nL,nrep);
gRel=[]; gOut=[];   % gate values (reliable vs outaged) collected at p=0.5
rng(7);
for i=1:nL
    p=pgrid(i);
    for r=1:nrep
        Xn = applyImpairment(D.XtestNN, 0, nBfix, 0, fv);
        [Pc, outMask] = outageMat(posTeN, p, sentinel);
        aG(i,r)=acc13(predict(netG,Xn,Pc),D);
        aR(i,r)=acc13(predict(netR,Xn,Pc),D);
        aB(i,r)=acc13(predict(netB,Xn),D);
        if abs(p-0.5)<1e-9
            g=predict(netG,Xn,Pc,Outputs="gate"); gpu=mean(g,2);  % per-UE mean gate
            gRel=[gRel; gpu(~outMask)]; gOut=[gOut; gpu(outMask)]; %#ok<AGROW>
        end
    end
end
Gm=mean(aG,2)'; Gs=std(aG,0,2)'; Rm=mean(aR,2)'; Rs=std(aR,0,2)'; Bm=mean(aB,2)';

fprintf('\n==== POSITION-RELIABILITY (nB=8 blockage, sweep outage rate) acc@K=13 ====\n');
fprintf('%8s | %-12s | %-12s | %-10s\n','outage p','GatedFus','PlainResid','RSRP-only');
for i=1:nL
    fprintf('%8.2f | %5.1f +/- %3.1f | %5.1f +/- %3.1f | %6.1f\n', pgrid(i), Gm(i),Gs(i), Rm(i),Rs(i), Bm(i));
end
fprintf('\nADAPTIVITY (gate value, p=0.5 draws): reliable-pos = %.3f   outaged-pos = %.3f   (ratio %.1fx)\n', ...
    mean(gRel), mean(gOut), mean(gRel)/max(mean(gOut),1e-6));

% ---- figures ----
figDir=fullfile(fileparts(base),'results','figures');
red=[0.85 0.1 0.1]; blue=[0 0.45 0.74]; gray=[0.4 0.4 0.4];
f=figure('Visible','off','Position',[100 100 720 460]); hold on; grid on;
errorbar(pgrid, Gm, Gs, '-o', LineWidth=2, Color=red, CapSize=4);
errorbar(pgrid, Rm, Rs, '-s', LineWidth=2, Color=blue, CapSize=4);
plot(pgrid, Bm, '--', LineWidth=1.5, Color=gray);
xlabel('Per-UE position-outage probability'); ylabel('Top-13 Accuracy (%)');
title('Variable Position Reliability (n_B=8 blockage)');
legend('Gated fusion','Plain residual (g=1)','RSRP-only (floor)','Location','northeast');
exportgraphics(f, fullfile(figDir,'novel_posreliability_acc.png'), Resolution=200); close(f);

f2=figure('Visible','off','Position',[100 100 560 460]); hold on; grid on;
bar([mean(gRel) mean(gOut)], 0.6, 'FaceColor',[0.2 0.5 0.2]);
set(gca,'XTick',[1 2],'XTickLabel',{'reliable position','outaged position'}); ylim([0 1]);
ylabel('Mean learned gate value'); title('Gate Adapts to Position Validity (p=0.5)');
exportgraphics(f2, fullfile(figDir,'novel_posreliability_gate.png'), Resolution=200); close(f2);
fprintf('Saved figures to %s\n', figDir);
disp('EXP_POSREL_DONE');

% ---- local functions ----
function out = ternary(cond, a, b), if cond, out=a; else, out=b; end, end
function a = acc13(pred, D)
    [~,ord]=sort(pred,2,'descend'); t=D.optTest(:); h=0;
    for n=1:D.Ntest, if any(ord(n,1:13)==t(n)), h=h+1; end; end
    a=100*h/D.Ntest;
end
function [Pc, mask] = outageMat(Pn, p, sentinel)
    N=size(Pn,1); mask=rand(N,1)<p; Pc=Pn; Pc(mask,:)=repmat(sentinel,nnz(mask),1);
end
