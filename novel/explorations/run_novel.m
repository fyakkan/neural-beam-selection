function R = run_novel(opts)
%run_novel  Train + evaluate the proposed multi-modal FUSION beam selector,
%           and compare it head-to-head against the reproduced baseline.
%
%   Reuses the baseline data pipeline (loadBeamData) and evaluator (evalTopK)
%   so the fusion net is trained/tested on IDENTICAL splits, seeds, and test
%   set, with the same KNN/Statistical/Random/Exhaustive references.
%
%   Options (name-value):
%       doTraining (true)   train from scratch; else load saved fusion net
%       saveNet    (true)   save to novel/nnBS_fusionNet.mat
%       maxEpochs  (300)
%       seed       (1)
%
%   Usage (headless):  matlab -batch "run_novel"

    arguments
        opts.doTraining (1,1) logical = true
        opts.saveNet    (1,1) logical = true
        opts.maxEpochs  (1,1) double  = 300
        opts.seed       (1,1) double  = 1
    end

    here     = fileparts(mfilename('fullpath'));
    repo     = fileparts(here);
    baseDir  = fullfile(repo, 'baseline');
    dataDir  = fullfile(baseDir, 'data');
    figDir   = fullfile(repo, 'results', 'figures');
    metDir   = fullfile(repo, 'results', 'metrics');
    addpath(here, baseDir, fullfile(baseDir,'helpers'));
    if ~exist(figDir,'dir'), mkdir(figDir); end
    if ~exist(metDir,'dir'), mkdir(metDir); end
    netFile = fullfile(here, 'nnBS_fusionNet.mat');

    % ---- data (identical to baseline) ----
    fprintf('Loading data from %s ...\n', dataDir);
    D = loadBeamData(dataDir);
    fprintf('  beam pairs=%d  RSRP inputs=%d  pos inputs=%d  train=%d  val=%d  test=%d\n', ...
        D.NumBeamPairs, D.numSampled, size(D.posTrain,2), D.NtrainNN, D.NvalNN, D.Ntest);

    % ---- fusion network ----
    if opts.doTraining
        rng(opts.seed);
        net = buildFusionNet(D.numSampled, size(D.posTrain,2), D.NumBeamPairs);

        % Multi-input data must be combined datastores: (rsrp, pos) inputs + target.
        dsTrain = combine(arrayDatastore(D.XtrainNN), arrayDatastore(D.posTrain), arrayDatastore(D.TtrainNN));
        dsVal   = combine(arrayDatastore(D.XvalNN),   arrayDatastore(D.posVal),   arrayDatastore(D.TvalNN));

        options = trainingOptions("adam", ...
            MaxEpochs          = opts.maxEpochs, ...
            MiniBatchSize      = 512, ...
            InitialLearnRate   = 1e-3, ...
            LearnRateSchedule  = "piecewise", ...
            LearnRateDropFactor= 0.5, ...
            LearnRateDropPeriod= 80, ...
            ValidationData     = dsVal, ...
            ValidationFrequency= 50, ...
            ValidationPatience = 20, ...
            OutputNetwork      = "best-validation-loss", ...
            Shuffle            = "every-epoch", ...
            ExecutionEnvironment = "cpu", ...
            Plots              = "none", ...
            Verbose            = true, ...
            VerboseFrequency   = 200);

        fprintf('Training fusion net (%d epochs) ...\n', opts.maxEpochs);
        t0 = tic;
        net = trainnet(dsTrain, net, "mse", options);
        trainTime = toc(t0);
        fprintf('  done in %.1f s\n', trainTime);
        if opts.saveNet
            save(netFile, 'net'); fprintf('  saved net -> %s\n', netFile);
        end
    else
        fprintf('Loading saved fusion net from %s ...\n', netFile);
        L = load(netFile); net = L.net; trainTime = NaN;
    end

    % ---- predict + evaluate (same evaluator as baseline) ----
    predFusion = predict(net, D.XtestNN, D.posTest);     % Ntest x NumBeamPairs
    R = evalTopK(predFusion, D);
    R.summary.trainTime_s = trainTime;
    R.summary.method = "fusion";

    % ---- load baseline for comparison ----
    baseRes = fullfile(metDir,'baseline_results.mat');
    haveBase = exist(baseRes,'file');
    if haveBase, B = load(baseRes); B = B.R; else, B = []; end

    % ---- figures + metrics ----
    makeComparisonFigures(R, B, figDir);
    save(fullfile(metDir,'novel_results.mat'), 'R');
    writeComparison(R, B, fullfile(metDir,'novel_metrics.json'));

    % ---- console summary ----
    s = R.summary;
    fprintf('\n==================== FUSION (NOVEL) SUMMARY ====================\n');
    fprintf('Top-K reaches 90%% at K = %s   (95%% at K = %s)\n', num2str(s.K90_Neural), num2str(s.K95_Neural));
    fprintf('Top-K acc @K=13: %.1f%%   @K=8: %.1f%%   @K=30: %.1f%%\n', ...
        s.acc_NN_at_K13, R.accNeural(min(8,end)), s.acc_NN_at_K30);
    fprintf('Avg RSRP @K=13: %.2f dB  (optimal %.2f dB)\n', s.rsrp_NN_at_K13, s.rsrp_optimal_dB);
    if haveBase
        fprintf('--- vs baseline (RSRP-only NN) ---\n');
        fprintf('  K@90%%:    fusion %d   baseline %d   (lower is better)\n', s.K90_Neural, B.summary.K90_Neural);
        for k = [5 8 13 20 30]
            fprintf('  acc@K=%2d: fusion %.1f%%   baseline %.1f%%   (+%.1f pts)\n', ...
                k, R.accNeural(k), B.accNeural(k), R.accNeural(k)-B.accNeural(k));
        end
        fprintf('  RSRP@K=13: fusion %.2f   baseline %.2f dB\n', R.rsrpNeural(13), B.rsrpNeural(13));
    end
    fprintf('===============================================================\n');
end

% ----------------------------------------------------------------------
function makeComparisonFigures(R, B, figDir)
    K = R.K;

    f1 = figure('Visible','off','Position',[100 100 720 500]);
    hold on; grid on;
    plot(K, R.accNeural, '-o', LineWidth=2, MarkerIndices=1:3:numel(K), Color=[0.85 0.1 0.1]);
    if ~isempty(B)
        plot(K, B.accNeural, '-*', LineWidth=1.8, MarkerIndices=1:3:numel(K), Color=[0 0.45 0.74]);
        plot(K, B.accKNN, '--s', LineWidth=1.3, MarkerIndices=1:4:numel(K), Color=[0.9 0.6 0]);
    end
    yline(90,':','90%');
    xlabel('K'); ylabel('Top-K Accuracy (%)');
    title('Top-K Accuracy: Proposed Fusion vs Baseline');
    if ~isempty(B)
        legend('Fusion (RSRP+Pos)','Baseline NN (RSRP)','KNN (Pos)','Location','southeast');
    else
        legend('Fusion (RSRP+Pos)','Location','southeast');
    end
    xlim([1 numel(K)]); ylim([0 100]);
    exportgraphics(f1, fullfile(figDir,'novel_topk_accuracy.png'), Resolution=200);
    close(f1);

    f2 = figure('Visible','off','Position',[100 100 720 500]);
    hold on; grid on;
    plot(K, R.rsrpNeural, '-o', LineWidth=2, MarkerIndices=1:3:numel(K), Color=[0.85 0.1 0.1]);
    if ~isempty(B)
        plot(K, B.rsrpNeural, '-*', LineWidth=1.8, MarkerIndices=1:3:numel(K), Color=[0 0.45 0.74]);
        plot(K, B.rsrpKNN, '--s', LineWidth=1.3, MarkerIndices=1:4:numel(K), Color=[0.9 0.6 0]);
        plot(K, B.rsrpOptimal, '-k', LineWidth=1.2);
    end
    xlabel('K'); ylabel('Average RSRP (dB)');
    title('Average RSRP: Proposed Fusion vs Baseline');
    if ~isempty(B)
        legend('Fusion (RSRP+Pos)','Baseline NN (RSRP)','KNN (Pos)','Exhaustive (optimal)','Location','southeast');
    else
        legend('Fusion (RSRP+Pos)','Location','southeast');
    end
    xlim([1 numel(K)]);
    exportgraphics(f2, fullfile(figDir,'novel_avg_rsrp.png'), Resolution=200);
    close(f2);
    fprintf('Saved comparison figures to %s\n', figDir);
end

function writeComparison(R, B, jsonFile)
    out = struct();
    out.fusion = R.summary;
    out.fusion.accNeural  = R.accNeural;
    out.fusion.rsrpNeural = R.rsrpNeural;
    out.K = R.K;
    if ~isempty(B)
        out.baseline = B.summary;
        out.baseline.accNeural  = B.accNeural;
        out.baseline.rsrpNeural = B.rsrpNeural;
        ks = [5 8 13 20 30];
        gains = arrayfun(@(k) R.accNeural(k)-B.accNeural(k), ks);
        out.comparison.K_points = ks;
        out.comparison.acc_gain_pts = gains;
        out.comparison.K90_fusion   = R.summary.K90_Neural;
        out.comparison.K90_baseline = B.summary.K90_Neural;
    end
    fid = fopen(jsonFile,'w');
    fwrite(fid, jsonencode(out, PrettyPrint=true));
    fclose(fid);
    fprintf('Saved comparison metrics to %s\n', jsonFile);
end
