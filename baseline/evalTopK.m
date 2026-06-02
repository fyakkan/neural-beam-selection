function R = evalTopK(predNN, D, randSeed)
%evalTopK  Top-K accuracy and average-RSRP curves for all beam-selection methods.
%
%   R = evalTopK(PREDNN, D, RANDSEED) evaluates, for K = 1..NumBeamPairs:
%     - Neural Network : ranks beams by the predicted RSRP profile PREDNN
%                        (Ntest x NumBeamPairs).
%     - KNN            : recommends beams of the K nearest training UEs (by position).
%     - Statistical    : the K most frequently-optimal beams in the test set.
%     - Random         : K random beams (per test UE).
%     - Exhaustive     : optimal beam over all 70 (RSRP upper bound; K-independent).
%
%   "Top-K accuracy" = fraction of test UEs whose TRUE optimal beam is among the
%   K recommended beams. "Average RSRP" at K = mean over test UEs of the best
%   ACTUAL RSRP among the K recommended beams (locations whose best-of-K is -Inf
%   contribute 0, matching the official example).
%
%   Returns a struct of 1xK curves plus a .summary with headline numbers.

    arguments
        predNN double
        D struct
        randSeed (1,1) double = 111
    end

    NB   = D.NumBeamPairs;
    Nte  = D.Ntest;
    trueOpt = D.optTest(:);            % Nte x 1, true best beam index
    rsrp = D.rsrpTest;                 % NB x Nte, ACTUAL (un-floored) RSRP

    % ---- precompute per-method beam ORDERINGS (Nte x NB, best first) ----
    % Neural network: descending predicted RSRP
    [~, ordNN] = sort(predNN, 2, 'descend');                 % Nte x NB

    % Statistical: descending test-set optimality frequency (same order for all UEs)
    statCount = accumarray(trueOpt, 1, [NB 1]);
    [~, statOrder] = sort(statCount, 'descend');             % NB x 1
    ordStat = repmat(statOrder.', Nte, 1);                   % Nte x NB

    % KNN: nearest training UEs by position -> their optimal beams (may repeat)
    knnIdx  = knnsearch(D.posTrain, D.posTest, 'K', min(NB, size(D.posTrain,1))); % Nte x NB
    ordKNN  = D.optTrain(knnIdx);                            % Nte x NB beam indices
    if size(ordKNN,2) < NB    % pad if fewer training points than NB (not expected)
        ordKNN(:, end+1:NB) = ordKNN(:, end);
    end

    % Random: a random permutation of all beams per test UE
    rng(randSeed);
    ordRand = zeros(Nte, NB);
    for n = 1:Nte
        ordRand(n,:) = randperm(NB);
    end

    % ---- Top-K ACCURACY (vectorised via "rank of the true beam") ----
    accNeural = accFromOrder(ordNN,  trueOpt, NB, Nte);
    accStat   = accFromOrder(ordStat, trueOpt, NB, Nte);
    accRandom = accFromOrder(ordRand, trueOpt, NB, Nte);
    accKNN    = accKNNfromOrder(ordKNN, trueOpt, NB, Nte);   % set membership (beams repeat)

    % ---- Average RSRP (cumulative max of ACTUAL RSRP along each ordering) ----
    rsrpNeural = rsrpFromOrder(ordNN,  rsrp, Nte);
    rsrpStat   = rsrpFromOrder(ordStat, rsrp, Nte);
    rsrpRandom = rsrpFromOrder(ordRand, rsrp, Nte);
    rsrpKNN    = rsrpFromOrder(ordKNN,  rsrp, Nte);

    % Exhaustive optimal (best beam over all NB), K-independent
    bestPer = max(rsrp, [], 1);                 % 1 x Nte (may be -Inf if fully blocked)
    rsrpOpt = sum(bestPer(isfinite(bestPer))) / Nte;
    rsrpOptimal = repmat(rsrpOpt, 1, NB);

    % ---- pack ----
    R = struct();
    R.K = 1:NB;
    R.accNeural = accNeural;  R.accKNN = accKNN;  R.accStatistic = accStat;  R.accRandom = accRandom;
    R.rsrpNeural = rsrpNeural; R.rsrpKNN = rsrpKNN; R.rsrpStatistic = rsrpStat;
    R.rsrpRandom = rsrpRandom; R.rsrpOptimal = rsrpOptimal;

    % ---- headline summary ----
    s = struct();
    s.NumBeamPairs   = NB;
    s.numSampled     = D.numSampled;
    s.Ntest          = Nte;
    s.K90_Neural     = firstKreaching(accNeural, 90);
    s.K95_Neural     = firstKreaching(accNeural, 95);
    s.acc_NN_at_K13  = accAt(accNeural, 13);
    s.acc_KNN_at_K13 = accAt(accKNN, 13);
    s.acc_NN_at_K30  = accAt(accNeural, 30);
    if ~isnan(s.K90_Neural)
        s.overheadReduction_pct_at_K90 = 100*(1 - s.K90_Neural/NB);
    else
        s.overheadReduction_pct_at_K90 = NaN;
    end
    s.rsrp_optimal_dB = rsrpOpt;
    s.rsrp_NN_at_K13  = rsrpNeural(min(13,NB));
    s.rsrp_KNN_at_K13 = rsrpKNN(min(13,NB));
    R.summary = s;
end

% ----------------------------------------------------------------------
function acc = accFromOrder(ord, trueOpt, NB, Nte)
    % rank (1=best) of the true beam within each UE's ordering, then CDF over K
    rankTrue = zeros(Nte,1);
    for n = 1:Nte
        r = find(ord(n,:) == trueOpt(n), 1);
        if isempty(r), r = NB; end
        rankTrue(n) = r;
    end
    acc = arrayfun(@(k) 100*mean(rankTrue <= k), 1:NB);
end

function acc = accKNNfromOrder(ord, trueOpt, NB, Nte)
    % KNN recommends a SET of beams (with repeats); first K neighbours.
    % Success at K if the true beam appears among the first K neighbours.
    % Default Inf => UEs whose true beam is absent from all neighbours never
    % count as a hit (so KNN need not reach 100% at K = NB, as expected).
    firstHit = inf(Nte,1);
    for n = 1:Nte
        h = find(ord(n,:) == trueOpt(n), 1);
        if ~isempty(h), firstHit(n) = h; end
    end
    acc = arrayfun(@(k) 100*mean(firstHit <= k), 1:NB);
end

function curve = rsrpFromOrder(ord, rsrp, Nte)
    % Best ACTUAL RSRP among the first K selected beams, averaged over UEs.
    NB = size(ord,2);
    A  = zeros(Nte, NB);
    for n = 1:Nte
        A(n,:) = rsrp(ord(n,:), n);          % actual RSRP along this UE's ordering
    end
    cmax = cummax(A, 2);                      % best-so-far as K grows
    cmax(~isfinite(cmax)) = 0;                % -Inf (all selected blocked) -> contribute 0
    curve = sum(cmax, 1) / Nte;
end

function k = firstKreaching(acc, thr)
    idx = find(acc >= thr, 1);
    if isempty(idx), k = NaN; else, k = idx; end
end

function a = accAt(acc, k)
    k = min(k, numel(acc));
    a = acc(k);
end
