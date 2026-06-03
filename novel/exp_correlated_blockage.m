% ADDITIONAL ROBUSTNESS STUDY (not the main result): correlated / contiguous
% beam blockage. The main result drops nB of the 14 sampled beams i.i.d. at
% random; here we instead drop a CONTIGUOUS run of nB sampled beams (a single
% physical obstruction occludes a contiguous angular block), and test whether the
% gated-fusion advantage persists. Models are the SAME i.i.d.-trained nets
% (novel/gated_nets.mat) -- this is a test-time distribution-shift robustness check.
base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
NB=D.NumBeamPairs; fv=D.floorVal; Ns=D.numSampled;   % Ns=14 sampled beams
L = load(fullfile(nov,'gated_nets.mat')); netB=L.netB; netF=L.netF;

figDir = fullfile(fileparts(base),'results','figures');
levels=[0 2 4 6 8 10 12]; nrep=10; nL=numel(levels);
aB=zeros(nL,nrep); aF=zeros(nL,nrep);
rng(7);
for i=1:nL
    nb=levels(i);
    for r=1:nrep
        Xn = dropContiguous(D.XtestNN, nb, fv);   % contiguous run of nb sampled beams
        aB(i,r)=acc13(predict(netB,Xn),D);
        aF(i,r)=acc13(predict(netF,Xn,D.posTest),D);
    end
end
Bm=mean(aB,2)'; Bs=std(aB,0,2)'; Fm=mean(aF,2)'; Fs=std(aF,0,2)';

fprintf('\n==== CONTIGUOUS blockage: acc@K=13 (mean +/- std over %d draws) ====\n', nrep);
fprintf('%9s | %-13s | %-13s | gain\n','#blocked','RSRP-only','GatedFus');
for i=1:nL
    fprintf('%9d | %5.1f +/- %3.1f | %5.1f +/- %3.1f | %+.1f\n', levels(i), Bm(i),Bs(i), Fm(i),Fs(i), Fm(i)-Bm(i));
end
fprintf('LaTeX rows (mean$\\pm$std):\n');
fprintf('  RSRP-only & %s \\\\\n', latexRow(Bm,Bs));
fprintf('  Gated fusion & %s \\\\\n', latexRow(Fm,Fs));
fprintf('  Gain (pts) & %s \\\\\n', strjoin(arrayfun(@(g) sprintf('$%+.1f$',g), Fm-Bm,'uni',0),' & '));

% figure
red=[0.85 0.1 0.1]; blue=[0 0.45 0.74];
f=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
errorbar(levels, Fm, Fs, '-o', LineWidth=2, Color=red,  CapSize=4);
errorbar(levels, Bm, Bs, '-*', LineWidth=2, Color=blue, CapSize=4);
xlabel('Number of contiguously blocked input beams (of 14)'); ylabel('Top-13 Accuracy (%)');
title('Robustness to Contiguous (Correlated) Blockage (mean \pm std)');
legend('Gated Fusion (RSRP+Pos)','RSRP-only','Location','northeast');
exportgraphics(f, fullfile(figDir,'novel_contiguous_blockage.png'), Resolution=200); close(f);
fprintf('Saved figure -> %s/novel_contiguous_blockage.png\n', figDir);
disp('EXP_CORR_BLOCK_DONE');

% ---- local functions ----
function X = dropContiguous(X, nb, fv)
    % Set a CONTIGUOUS run of nb of the M sampled beams (random start, wrap) to fv,
    % independently per observation. nb=0 -> unchanged.
    [N,M]=size(X);
    if nb<=0, return; end
    nb=min(nb,M);
    for n=1:N
        s=randi(M);
        idx=mod(s-1+(0:nb-1),M)+1;   % contiguous run on the ring of M sampled beams
        X(n,idx)=fv;
    end
end

function a = acc13(pred, D)
    [~,ord]=sort(pred,2,'descend'); t=D.optTest(:); h=0;
    for n=1:D.Ntest, if any(ord(n,1:13)==t(n)), h=h+1; end; end
    a=100*h/D.Ntest;
end

function s = latexRow(m, sd)
    parts = arrayfun(@(a,b) sprintf('$%.1f_{\\pm%.1f}$', a, b), m, sd, 'uni', 0);
    s = strjoin(parts, ' & ');
end
