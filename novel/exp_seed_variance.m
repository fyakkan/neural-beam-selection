function R = exp_seed_variance(seeds, maxEpochs)
%exp_seed_variance  SEED-VARIANCE study for Table III (paper revision A1).
%
%   The per-draw std reported in Table III captures only test-time blockage-draw
%   variance; it does NOT capture training-seed / weight-initialization /
%   augmentation-stream variance, because the headline numbers come from ONE
%   trained RSRP-only net and ONE trained gated-fusion net. This script retrains
%   BOTH models across several random seeds, recomputes the i.i.d. blockage sweep
%   per seed, and reports the gain (gated - RSRP-only) as mean +/- std ACROSS
%   SEEDS at each blockage level, demonstrating the gain survives init noise.
%
%   To isolate seed variance, every seed is evaluated on the SAME fixed set of 10
%   i.i.d. blockage draws (rng=7), so the across-seed spread reflects training
%   randomness only. Each per-seed accuracy is first averaged over the 10 draws.
%
%   Usage (headless):  matlab -batch "exp_seed_variance"
%                      matlab -batch "exp_seed_variance([1 2 3],300)"

    if nargin < 1 || isempty(seeds),     seeds = 1:5;  end
    if nargin < 2 || isempty(maxEpochs), maxEpochs = 300; end

    here    = fileparts(mfilename('fullpath'));
    repo    = fileparts(here);
    baseDir = fullfile(repo,'baseline');
    dataDir = fullfile(baseDir,'data');
    metDir  = fullfile(repo,'results','metrics');
    addpath(here, baseDir, fullfile(baseDir,'helpers'));
    if ~exist(metDir,'dir'), mkdir(metDir); end

    D = loadBeamData(dataDir);
    NB = D.NumBeamPairs; fv = D.floorVal;
    blevels = 0:1:12;  K = 13;  nrep = 10;  nL = numel(blevels);
    nS = numel(seeds);

    accB_seed = zeros(nS,nL);   % per-seed accuracy (mean over the 10 fixed draws)
    accF_seed = zeros(nS,nL);

    topt = trainingOptions("adam", MaxEpochs=maxEpochs, MiniBatchSize=512, ...
        InitialLearnRate=1e-3, LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, ...
        LearnRateDropPeriod=80, ExecutionEnvironment="cpu", Plots="none", Verbose=false);

    for si = 1:nS
        s = seeds(si);
        fprintf('\n===== SEED %d  (%d/%d) =====\n', s, si, nS); t0 = tic;

        % SAME impairment-aware augmentation as run_gated.m (blockage 0..8 + light noise)
        augOne = @(x) applyImpairment(x, 2*rand, randi([0 8]), 0, fv);
        augR = @(c) {augOne(c{1}), c{2}};
        augF = @(c) {augOne(c{1}), c{2}, c{3}};

        rng(s);
        dsR  = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.TtrainNN)), augR);
        netB = trainnet(dsR, buildRegressionNet(D.numSampled,NB,96), "mse", topt);

        rng(s);
        dsF  = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.posTrain),arrayDatastore(D.TtrainNN)), augF);
        netF = trainnet(dsF, buildGatedFusionNet(D.numSampled,size(D.posTrain,2),NB), "mse", topt);

        % fixed i.i.d. blockage draws (identical for every seed)
        rng(7);
        Xstore = cell(nL,1);
        for i = 1:nL
            Xi = zeros(D.Ntest, D.numSampled, nrep);
            for r = 1:nrep, Xi(:,:,r) = applyImpairment(D.XtestNN, 0, blevels(i), 0, fv); end
            Xstore{i} = Xi;
        end

        for i = 1:nL
            aB = zeros(1,nrep); aF = zeros(1,nrep);
            for r = 1:nrep
                Xn    = Xstore{i}(:,:,r);
                aB(r) = acc13(predict(netB, Xn), D, K);
                aF(r) = acc13(predict(netF, Xn, D.posTest), D, K);
            end
            accB_seed(si,i) = mean(aB);
            accF_seed(si,i) = mean(aF);
        end

        fprintf('  clean: RSRP=%.1f Fusion=%.1f | nB=6: %+.1f | nB=8: %+.1f | nB=10: %+.1f  (%.0fs)\n', ...
            accB_seed(si,1), accF_seed(si,1), ...
            accF_seed(si,7)-accB_seed(si,7), accF_seed(si,9)-accB_seed(si,9), ...
            accF_seed(si,11)-accB_seed(si,11), toc(t0));

        % checkpoint after every seed (survives interruption)
        save(fullfile(metDir,'seed_variance_partial.mat'),'seeds','blevels','accB_seed','accF_seed','si');
    end

    gain_seed = accF_seed - accB_seed;
    accB_mean = mean(accB_seed,1);  accB_sd = std(accB_seed,0,1);
    accF_mean = mean(accF_seed,1);  accF_sd = std(accF_seed,0,1);
    gain_mean = mean(gain_seed,1);  gain_sd = std(gain_seed,0,1);

    R = struct('seeds',seeds,'blevels',blevels,'K',K,'nrep',nrep,'nseeds',nS, ...
        'accB_seed',accB_seed,'accF_seed',accF_seed,'gain_seed',gain_seed, ...
        'accB_mean',accB_mean,'accB_sd',accB_sd,'accF_mean',accF_mean,'accF_sd',accF_sd, ...
        'gain_mean',gain_mean,'gain_sd',gain_sd);
    save(fullfile(metDir,'seed_variance.mat'),'R');
    fid = fopen(fullfile(metDir,'seed_variance.json'),'w'); fwrite(fid, jsonencode(R,PrettyPrint=true)); fclose(fid);

    % ---- LaTeX rows for Table III columns (0 2 4 6 8 10 12) ----
    cols = ismember(blevels,[0 2 4 6 8 10 12]);
    fprintf('\n===== SEED-VARIANCE TABLE III  (acc@K=13, %%, mean+/-std ACROSS %d seeds) =====\n', nS);
    fprintf('cols = 0 2 4 6 8 10 12\n');
    fprintf('  RSRP-only    & %s \\\\\n', lrow(accB_mean(cols), accB_sd(cols)));
    fprintf('  Gated fusion & %s \\\\\n', lrow(accF_mean(cols), accF_sd(cols)));
    fprintf('  Gain (pts)   & %s \\\\\n', lrow(gain_mean(cols), gain_sd(cols)));
    fprintf('seeds = %s\n', mat2str(seeds));
    fprintf('per-seed clean RSRP acc: %s\n', mat2str(round(accB_seed(:,1)',1)));
    fprintf('per-seed clean Fusion acc: %s\n', mat2str(round(accF_seed(:,1)',1)));
    disp('EXP_SEED_VARIANCE_DONE');
end

% ---------------------------------------------------------------------------
function a = acc13(pred, D, K)
    [~,ord] = sort(pred, 2, 'descend'); t = D.optTest(:); h = 0;
    for n = 1:D.Ntest, if any(ord(n,1:K) == t(n)), h = h + 1; end, end
    a = 100*h/D.Ntest;
end

function s = lrow(m, sd)
    s = strjoin(arrayfun(@(a,b) sprintf('\\ms{%.1f}{%.1f}', a, b), m, sd, 'uni', 0), ' & ');
end
