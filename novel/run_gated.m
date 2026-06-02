function R = run_gated(opts)
%run_gated  Train + evaluate the proposed GATED multi-modal fusion beam selector,
%           and compare its robustness to beam blockage against an RSRP-only model.
%
%   Both models are trained impairment-aware with the SAME blockage augmentation
%   (0..bMax of the sampled beams dropped, plus light noise); they differ only in
%   that the proposed model also ingests UE position through a gated residual path.
%   Evaluated on the IDENTICAL data/splits/seeds as the baseline (loadBeamData,
%   evalTopK). The headline result is robustness to mmWave beam blockage at no
%   clean-data cost.
%
%   Options: doTraining(true) saveNet(true) maxEpochs(300) seed(1)
%            bMaxTrain(8) sigMaxTrain(2) nrep(10) repLevel(8)
%   Usage (headless):  matlab -batch "run_gated"

    arguments
        opts.doTraining (1,1) logical = true
        opts.saveNet    (1,1) logical = true
        opts.maxEpochs  (1,1) double  = 300
        opts.seed       (1,1) double  = 1
        opts.bMaxTrain  (1,1) double  = 8
        opts.sigMaxTrain(1,1) double  = 2
        opts.nrep       (1,1) double  = 10
        opts.repLevel   (1,1) double  = 8
    end

    here    = fileparts(mfilename('fullpath'));
    repo    = fileparts(here);
    baseDir = fullfile(repo,'baseline');
    dataDir = fullfile(baseDir,'data');
    figDir  = fullfile(repo,'results','figures');
    metDir  = fullfile(repo,'results','metrics');
    addpath(here, baseDir, fullfile(baseDir,'helpers'));
    if ~exist(figDir,'dir'), mkdir(figDir); end
    if ~exist(metDir,'dir'), mkdir(metDir); end
    netFile = fullfile(here,'gated_nets.mat');

    D = loadBeamData(dataDir);
    NB = D.NumBeamPairs; fv = D.floorVal;

    % ---- impairment-aware training (blockage + light noise) ----
    if opts.doTraining
        sm = opts.sigMaxTrain; bm = opts.bMaxTrain;
        augOne = @(x) applyImpairment(x, sm*rand, randi([0 bm]), 0, fv);
        augR = @(c) {augOne(c{1}), c{2}};
        augF = @(c) {augOne(c{1}), c{2}, c{3}};
        topt = trainingOptions("adam", MaxEpochs=opts.maxEpochs, MiniBatchSize=512, ...
            InitialLearnRate=1e-3, LearnRateSchedule="piecewise", LearnRateDropFactor=0.5, ...
            LearnRateDropPeriod=80, ExecutionEnvironment="cpu", Plots="none", Verbose=false);

        fprintf('Training RSRP-only (impairment-aware) ...\n'); t0=tic;
        rng(opts.seed);
        dsR = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.TtrainNN)), augR);
        netB = trainnet(dsR, buildRegressionNet(D.numSampled,NB,96), "mse", topt);
        fprintf('  %.1fs\n', toc(t0));

        fprintf('Training GATED fusion (impairment-aware) ...\n'); t0=tic;
        rng(opts.seed);
        dsF = transform(combine(arrayDatastore(D.XtrainNN),arrayDatastore(D.posTrain),arrayDatastore(D.TtrainNN)), augF);
        netF = trainnet(dsF, buildGatedFusionNet(D.numSampled,size(D.posTrain,2),NB), "mse", topt);
        fprintf('  %.1fs\n', toc(t0));
        if opts.saveNet, save(netFile,'netB','netF'); fprintf('  saved -> %s\n', netFile); end
    else
        L = load(netFile); netB = L.netB; netF = L.netF;
    end

    % ---- blockage sweep ----
    blevels = 0:1:12;  K = 13;
    accB = zeros(size(blevels)); accF = accB; rsB = accB; rsF = accB; gateMean = accB;
    rng(7);
    for i = 1:numel(blevels)
        nb = blevels(i);
        for r = 1:opts.nrep
            Xn = applyImpairment(D.XtestNN, 0, nb, 0, fv);
            pB = predict(netB, Xn);
            pF = predict(netF, Xn, D.posTest);
            [aB,rB] = nnAccRsrp(pB, D, K);
            [aF,rF] = nnAccRsrp(pF, D, K);
            accB(i)=accB(i)+aB/opts.nrep;  rsB(i)=rsB(i)+rB/opts.nrep;
            accF(i)=accF(i)+aF/opts.nrep;  rsF(i)=rsF(i)+rF/opts.nrep;
            g = predict(netF, Xn, D.posTest, Outputs="gate");
            gateMean(i) = gateMean(i) + mean(g(:))/opts.nrep;
        end
    end

    % position-only (KNN) reference + optimal (blockage-independent), via baseline evaluator
    R0 = evalTopK(predict(netB, D.XtestNN), D);
    accKNN13 = R0.accKNN(K);  rsrpKNN13 = R0.rsrpKNN(K);  rsrpOpt = R0.summary.rsrp_optimal_dB;

    % ---- representative blockage level: full Top-K curves ----
    rng(123);
    Xrep = applyImpairment(D.XtestNN, 0, opts.repLevel, 0, fv);
    Rrep_F = evalTopK(predict(netF, Xrep, D.posTest), D);
    Rrep_B = evalTopK(predict(netB, Xrep), D);

    % ---- figures ----
    makeGatedFigures(blevels, accB, accF, accKNN13, rsB, rsF, rsrpKNN13, rsrpOpt, ...
                     gateMean, Rrep_F, Rrep_B, opts.repLevel, K, figDir);

    % ---- metrics ----
    R = struct('blevels',blevels,'accRSRP',accB,'accFusion',accF,'rsrpRSRP',rsB,'rsrpFusion',rsF, ...
        'gateMean',gateMean,'accKNN13',accKNN13,'rsrpKNN13',rsrpKNN13,'rsrpOpt',rsrpOpt, ...
        'K',K,'repLevel',opts.repLevel);
    R.acc_clean_RSRP = accB(1); R.acc_clean_Fusion = accF(1);
    save(fullfile(metDir,'novel_results.mat'),'R');
    fid=fopen(fullfile(metDir,'novel_metrics.json'),'w'); fwrite(fid, jsonencode(R, PrettyPrint=true)); fclose(fid);

    % ---- console summary ----
    fprintf('\n============ GATED FUSION: BLOCKAGE ROBUSTNESS (acc@K=%d) ============\n', K);
    fprintf('%8s | %-9s | %-9s | gain\n','#blocked','RSRP-only','GatedFus');
    for i=1:numel(blevels)
        fprintf('%8d | %8.1f  | %8.1f  | %+.1f\n', blevels(i), accB(i), accF(i), accF(i)-accB(i));
    end
    fprintf('clean parity gain: %+.1f pts | mean gate: clean=%.2f, %d-blocked=%.2f\n', ...
        accF(1)-accB(1), gateMean(1), opts.repLevel, gateMean(blevels==opts.repLevel));
    fprintf('position-only (KNN) acc@K=%d = %.1f%% (blockage-independent)\n', K, accKNN13);
    fprintf('=====================================================================\n');
end

% ---------------------------------------------------------------------------
function [acc, rsrp] = nnAccRsrp(pred, D, K)
    % NN-only Top-K accuracy (%) and average actual RSRP (dB) at a single K
    Nte = D.Ntest; trueOpt = D.optTest(:); R = D.rsrpTest;
    [~, ord] = sort(pred, 2, 'descend');
    hit = 0; rs = 0;
    for n = 1:Nte
        topk = ord(n,1:K);
        if any(topk == trueOpt(n)), hit = hit + 1; end
        m = max(R(topk, n));
        if isfinite(m), rs = rs + m; end
    end
    acc = 100*hit/Nte;  rsrp = rs/Nte;
end

function makeGatedFigures(bl, accB, accF, accKNN, rsB, rsF, rsKNN, rsOpt, gateMean, Rf, Rb, repL, K, figDir)
    % Fig 1: accuracy vs blockage
    f1=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    plot(bl, accF, '-o', LineWidth=2, Color=[0.85 0.1 0.1]);
    plot(bl, accB, '-*', LineWidth=2, Color=[0 0.45 0.74]);
    yline(accKNN, '--', sprintf('KNN (position) %.0f%%',accKNN), Color=[0.9 0.6 0]);
    xlabel('Number of blocked input beams (of 14)'); ylabel(sprintf('Top-%d Accuracy (%%)',K));
    title('Robustness to Beam Blockage'); legend('Gated Fusion (RSRP+Pos)','RSRP-only','Location','northeast');
    exportgraphics(f1, fullfile(figDir,'novel_blockage_accuracy.png'), Resolution=200); close(f1);

    % Fig 2: avg RSRP vs blockage
    f2=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    plot(bl, rsF, '-o', LineWidth=2, Color=[0.85 0.1 0.1]);
    plot(bl, rsB, '-*', LineWidth=2, Color=[0 0.45 0.74]);
    yline(rsOpt,'-k',sprintf('Exhaustive %.1f dB',rsOpt));
    xlabel('Number of blocked input beams (of 14)'); ylabel(sprintf('Average RSRP at K=%d (dB)',K));
    title('Average RSRP vs Beam Blockage'); legend('Gated Fusion','RSRP-only','Location','southwest');
    exportgraphics(f2, fullfile(figDir,'novel_blockage_rsrp.png'), Resolution=200); close(f2);

    % Fig 3: Top-K curve at representative blockage
    f3=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    plot(Rf.K, Rf.accNeural, '-o', LineWidth=2, MarkerIndices=1:3:70, Color=[0.85 0.1 0.1]);
    plot(Rb.K, Rb.accNeural, '-*', LineWidth=2, MarkerIndices=1:3:70, Color=[0 0.45 0.74]);
    plot(Rf.K, Rf.accKNN, '--s', LineWidth=1.3, MarkerIndices=1:4:70, Color=[0.9 0.6 0]);
    yline(90,':','90%');
    xlabel('K'); ylabel('Top-K Accuracy (%)'); xlim([1 70]); ylim([0 100]);
    title(sprintf('Top-K Accuracy at %d/14 Beams Blocked', repL));
    legend('Gated Fusion','RSRP-only','KNN (position)','Location','southeast');
    exportgraphics(f3, fullfile(figDir,'novel_blockage_topk.png'), Resolution=200); close(f3);

    % Fig 4: gate behaviour (mechanism). Honest: the learned gate is ~constant
    % (~0.78); robustness comes from the always-on residual position path, not
    % from an adaptive gate. Fixed [0,1] axis so the near-constancy is visible.
    f4=figure('Visible','off','Position',[100 100 720 460]); grid on; hold on;
    plot(bl, gateMean, '-d', LineWidth=2, Color=[0.2 0.5 0.2]);
    ylim([0 1]);
    xlabel('Number of blocked input beams (of 14)'); ylabel('Mean learned gate value');
    title(sprintf('Learned Gate is Nearly Constant (\\approx%.2f)', mean(gateMean)));
    text(0.5, mean(gateMean)+0.06, sprintf('range %.3f-%.3f over all blockage levels', min(gateMean), max(gateMean)));
    exportgraphics(f4, fullfile(figDir,'novel_gate_behavior.png'), Resolution=200); close(f4);
    fprintf('Saved 4 figures to %s\n', figDir);
end
