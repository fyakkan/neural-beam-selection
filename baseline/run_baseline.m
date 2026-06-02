function R = run_baseline(opts)
%run_baseline  Reproduce the 5G NR beam-selection REGRESSION benchmark (Task 1).
%
%   R = run_baseline() loads the official pre-recorded data (15k/500), trains
%   the RSRP-regression network (14 downsampled RSRP -> 70-element RSRP profile),
%   evaluates Top-K accuracy and average RSRP against the KNN / Statistical /
%   Random / Exhaustive benchmarks, saves the two comparison figures and a
%   metrics JSON, and prints a summary.
%
%   Options (name-value):
%       doTraining (true)   train from scratch; if false, load saved net
%       saveNet    (true)   save the trained net to baseline/nnBS_regNet.mat
%       maxEpochs  (300)
%       hiddenWidth(96)
%       seed       (1)      global rng seed for training reproducibility
%
%   Usage (headless):  matlab -batch "run_baseline"

    arguments
        opts.doTraining  (1,1) logical = true
        opts.saveNet     (1,1) logical = true
        opts.maxEpochs   (1,1) double  = 300
        opts.hiddenWidth (1,1) double  = 96
        opts.seed        (1,1) double  = 1
    end

    % ---- paths (resolve relative to this file) ----
    here    = fileparts(mfilename('fullpath'));
    repo    = fileparts(here);
    dataDir = fullfile(here, 'data');
    figDir  = fullfile(repo, 'results', 'figures');
    metDir  = fullfile(repo, 'results', 'metrics');
    addpath(here, fullfile(here,'helpers'));
    if ~exist(figDir,'dir'), mkdir(figDir); end
    if ~exist(metDir,'dir'), mkdir(metDir); end
    netFile = fullfile(here, 'nnBS_regNet.mat');

    % ---- data ----
    fprintf('Loading data from %s ...\n', dataDir);
    D = loadBeamData(dataDir);
    fprintf('  beam pairs=%d  sampled inputs=%d  train=%d  val=%d  test=%d\n', ...
        D.NumBeamPairs, D.numSampled, D.NtrainNN, D.NvalNN, D.Ntest);

    % ---- network ----
    if opts.doTraining
        rng(opts.seed);
        layers = buildRegressionNet(D.numSampled, D.NumBeamPairs, opts.hiddenWidth);
        options = trainingOptions("adam", ...
            MaxEpochs          = opts.maxEpochs, ...
            MiniBatchSize      = 512, ...
            InitialLearnRate   = 1e-3, ...
            LearnRateSchedule  = "piecewise", ...
            LearnRateDropFactor= 0.5, ...
            LearnRateDropPeriod= 80, ...
            ValidationData     = {D.XvalNN, D.TvalNN}, ...
            ValidationFrequency= 50, ...
            ValidationPatience = 20, ...
            OutputNetwork      = "best-validation-loss", ...
            Shuffle            = "every-epoch", ...
            ExecutionEnvironment = "cpu", ...
            Plots              = "none", ...
            Verbose            = true, ...
            VerboseFrequency   = 200);

        fprintf('Training regression net (%d epochs, width=%d) ...\n', opts.maxEpochs, opts.hiddenWidth);
        t0 = tic;
        net = trainnet(D.XtrainNN, D.TtrainNN, layers, "mse", options);
        trainTime = toc(t0);
        fprintf('  done in %.1f s\n', trainTime);
        if opts.saveNet
            save(netFile, 'net', 'D');
            fprintf('  saved net -> %s\n', netFile);
        end
    else
        fprintf('Loading saved net from %s ...\n', netFile);
        L = load(netFile); net = L.net; trainTime = NaN;
    end

    % ---- predict + evaluate ----
    predNN = predict(net, D.XtestNN);            % Ntest x NumBeamPairs
    R = evalTopK(predNN, D);
    R.summary.trainTime_s = trainTime;
    R.summary.hiddenWidth = opts.hiddenWidth;
    R.summary.maxEpochs   = opts.maxEpochs;

    % ---- figures ----
    makeFigures(R, figDir);

    % ---- metrics ----
    save(fullfile(metDir,'baseline_results.mat'), 'R');
    writeMetricsJSON(R, fullfile(metDir,'baseline_metrics.json'));

    % ---- console summary ----
    s = R.summary;
    fprintf('\n==================== BASELINE SUMMARY ====================\n');
    fprintf('Beam pairs: %d   sampled inputs: %d   test UEs: %d\n', s.NumBeamPairs, s.numSampled, s.Ntest);
    fprintf('NN Top-K reaches 90%% at K = %s   (95%% at K = %s)\n', num2str(s.K90_Neural), num2str(s.K95_Neural));
    fprintf('   -> overhead reduction at K90: %.1f%% (vs sweeping all %d)\n', s.overheadReduction_pct_at_K90, s.NumBeamPairs);
    fprintf('Top-K accuracy @K=13:  NN=%.1f%%   KNN=%.1f%%\n', s.acc_NN_at_K13, s.acc_KNN_at_K13);
    fprintf('Top-K accuracy @K=30:  NN=%.1f%%\n', s.acc_NN_at_K30);
    fprintf('Avg RSRP (dB) @K=13:   NN=%.2f   KNN=%.2f   Optimal=%.2f\n', s.rsrp_NN_at_K13, s.rsrp_KNN_at_K13, s.rsrp_optimal_dB);
    fprintf('==========================================================\n');
end

% ----------------------------------------------------------------------
function makeFigures(R, figDir)
    K = R.K;

    f1 = figure('Visible','off','Position',[100 100 700 480]);
    hold on; grid on;
    plot(K, R.accNeural,    '-*', LineWidth=1.5);
    plot(K, R.accKNN,       '--o', LineWidth=1.5, MarkerIndices=1:4:numel(K));
    plot(K, R.accStatistic, '--s', LineWidth=1.5, MarkerIndices=1:4:numel(K));
    plot(K, R.accRandom,    '--d', LineWidth=1.5, MarkerIndices=1:4:numel(K));
    yline(90,':','90%');
    xlabel('K'); ylabel('Top-K Accuracy (%)');
    title('Top-K Accuracy: Regression NN vs Benchmarks');
    legend('Neural Network','KNN','Statistical Info','Random','Location','southeast');
    xlim([1 numel(K)]); ylim([0 100]);
    exportgraphics(f1, fullfile(figDir,'baseline_topk_accuracy.png'), Resolution=200);
    close(f1);

    f2 = figure('Visible','off','Position',[100 100 700 480]);
    hold on; grid on;
    plot(K, R.rsrpNeural,    '-*', LineWidth=1.5);
    plot(K, R.rsrpKNN,       '--o', LineWidth=1.5, MarkerIndices=1:4:numel(K));
    plot(K, R.rsrpStatistic, '--s', LineWidth=1.5, MarkerIndices=1:4:numel(K));
    plot(K, R.rsrpRandom,    '--d', LineWidth=1.5, MarkerIndices=1:4:numel(K));
    plot(K, R.rsrpOptimal,   '-k',  LineWidth=1.2);
    xlabel('K'); ylabel('Average RSRP (dB)');
    title('Average RSRP: Regression NN vs Benchmarks');
    legend('Neural Network','KNN','Statistical Info','Random','Exhaustive (optimal)','Location','southeast');
    xlim([1 numel(K)]);
    exportgraphics(f2, fullfile(figDir,'baseline_avg_rsrp.png'), Resolution=200);
    close(f2);

    fprintf('Saved figures to %s\n', figDir);
end

function writeMetricsJSON(R, jsonFile)
    out = R.summary;
    out.K              = R.K;
    out.accNeural      = R.accNeural;
    out.accKNN         = R.accKNN;
    out.accStatistic   = R.accStatistic;
    out.accRandom      = R.accRandom;
    out.rsrpNeural     = R.rsrpNeural;
    out.rsrpKNN        = R.rsrpKNN;
    out.rsrpStatistic  = R.rsrpStatistic;
    out.rsrpRandom     = R.rsrpRandom;
    out.rsrpOptimal    = R.rsrpOptimal;
    fid = fopen(jsonFile,'w');
    fwrite(fid, jsonencode(out, PrettyPrint=true));
    fclose(fid);
    fprintf('Saved metrics to %s\n', jsonFile);
end
