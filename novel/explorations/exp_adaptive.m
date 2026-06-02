base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
addpath(base, fullfile(base,'helpers'));
D = loadBeamData(fullfile(base,'data'));
L = load(fullfile(base,'nnBS_regNet.mat')); net = L.net;
NB = D.NumBeamPairs; Nte = D.Ntest;
trueOpt = D.optTest(:); rsrp = D.rsrpTest;          % 70 x Nte actual

pred = predict(net, D.XtestNN);                     % Nte x 70 (normalized predicted RSRP)
[predSorted, ord] = sort(pred, 2, 'descend');

% ---- FIXED-K baseline curve (accuracy vs K) ----
rankTrue = zeros(Nte,1);
for n=1:Nte, rankTrue(n) = find(ord(n,:)==trueOpt(n),1); end
accFixed = arrayfun(@(k) 100*mean(rankTrue<=k), 1:NB);
Kfix90 = find(accFixed>=90,1);

% ---- ADAPTIVE-K via nucleus (top-p) on softmax(pred/T) ----
fprintf('Fixed-K baseline: K@90 = %d  (acc@13=%.1f)\n', Kfix90, accFixed(13));
fprintf('\nAdaptive-K (stop when cumulative confidence >= tau):\n');
for T = [0.10 0.15 0.20]
    P = softmax((pred./T)')';                       % Nte x 70 along beams
    [Psort,~] = deal(sort(P,2,'descend'));
    cumP = cumsum(Psort,2);
    bestMeanK90 = inf; bestTau = NaN; bestAcc = NaN;
    for tau = [0.5:0.05:0.95 0.97 0.99 0.995 0.999]
        Kn = zeros(Nte,1);
        for n=1:Nte
            kk = find(cumP(n,:)>=tau,1); if isempty(kk), kk=NB; end
            Kn(n)=kk;
        end
        succ = rankTrue <= Kn;                       % true beam within adaptive set
        acc = 100*mean(succ); mK = mean(Kn);
        if acc>=90 && mK<bestMeanK90, bestMeanK90=mK; bestTau=tau; bestAcc=acc; end
    end
    if isfinite(bestMeanK90)
        fprintf('  T=%.2f: reach 90%% acc at MEAN K = %.2f (tau=%.3f, acc=%.1f) -> %.0f%% less overhead than fixed K=%d\n', ...
            T, bestMeanK90, bestTau, bestAcc, 100*(1-bestMeanK90/Kfix90), Kfix90);
    else
        fprintf('  T=%.2f: did not reach 90%% within cap\n', T);
    end
end

% Detailed Pareto for the best T to compare at matched MEAN overhead
T=0.15; P=softmax((pred./T)')'; [Psort,~]=deal(sort(P,2,'descend')); cumP=cumsum(Psort,2);
fprintf('\nMatched-overhead comparison (T=0.15):\n');
for tau = [0.6 0.7 0.8 0.9 0.95 0.99]
    Kn=zeros(Nte,1); for n=1:Nte, kk=find(cumP(n,:)>=tau,1); if isempty(kk),kk=NB; end; Kn(n)=kk; end
    acc=100*mean(rankTrue<=Kn); mK=mean(Kn);
    accFixedAtSameK = accFixed(max(1,round(mK)));
    fprintf('  tau=%.2f: meanK=%.2f  adaptiveAcc=%.1f%%  fixedAcc@K=%d=%.1f%%  (gain %+.1f pts)\n', ...
        tau, mK, acc, round(mK), accFixedAtSameK, acc-accFixedAtSameK);
end
disp('EXP_ADAPTIVE_DONE');
