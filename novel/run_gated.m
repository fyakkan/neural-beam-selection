function R = run_gated(opts)
%run_gated  Train + evaluate the proposed GATED multi-modal fusion beam selector,
%           and emit BOTH the blockage-robustness table (Table III) and the
%           architecture-ablation table (Table IV) from a SINGLE shared set of
%           blockage draws, so every cell shared between the tables is identical.
%
%   Models (all trained impairment-aware with the SAME blockage augmentation):
%     netB  RSRP-only            (baseline/buildRegressionNet, single input)
%     netF  gated residual fusion (proposed)            } loaded from gated_nets.mat
%     netC  concatenation fusion  (ablation)            } loaded from
%     netR  plain residual fusion (ablation, no gate)   } ablation_arch_nets.mat
%
%   Evaluated on the IDENTICAL data/splits/seeds as the baseline. Blockage results
%   are averaged over nrep random draws; std across draws is reported. The 10
%   blockage masks are pre-generated once (rng=7) and reused by every model, table,
%   and figure -> Table III and Table IV agree exactly on their shared cells.
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

    % ablation nets (concat, plain residual) for Table IV -- evaluated on the SAME draws
    arcFile = fullfile(here,'ablation_arch_nets.mat'); haveArc = isfile(arcFile);
    if haveArc, AA = load(arcFile); netC = AA.netC; netR = AA.netR; end

    % ---- ONE shared set of blockage draws used by every model/table/figure ----
    blevels = 0:1:12;  K = 13;  nrep = opts.nrep;  nL = numel(blevels);
    rng(7);
    Xstore = cell(nL,1);
    for i = 1:nL
        Xi = zeros(D.Ntest, D.numSampled, nrep);
        for r = 1:nrep
            Xi(:,:,r) = applyImpairment(D.XtestNN, 0, blevels(i), 0, fv);  % only RNG consumer
        end
        Xstore{i} = Xi;
    end

    % ---- evaluate each model on the shared draws (no RNG consumed here) ----
    [accB_all, rsB_all] = evalModelAcc(netB, Xstore, D, K, false);
    [accF_all, rsF_all] = evalModelAcc(netF, Xstore, D, K, true);
    accB=mean(accB_all,2)';  accB_std=std(accB_all,0,2)';
    accF=mean(accF_all,2)';  accF_std=std(accF_all,0,2)';
    rsB =mean(rsB_all,2)';   rsB_std =std(rsB_all,0,2)';
    rsF =mean(rsF_all,2)';   rsF_std =std(rsF_all,0,2)';
    if haveArc
        accC_all = evalModelAcc(netC, Xstore, D, K, true);
        accR_all = evalModelAcc(netR, Xstore, D, K, true);
        accC=mean(accC_all,2)'; accC_std=std(accC_all,0,2)';
        accR=mean(accR_all,2)'; accR_std=std(accR_all,0,2)';
    end

    % gate values (netF only) on the shared draws
    gate_all = zeros(nL,nrep);
    for i=1:nL, for r=1:nrep
        g = predict(netF, Xstore{i}(:,:,r), D.posTest, Outputs="gate");
        gate_all(i,r)=mean(g(:));
    end, end
    gateMean=mean(gate_all,2)';

    % position-only (KNN) reference + optimal (blockage-independent)
    R0 = evalTopK(predict(netB, D.XtestNN), D);
    accKNN13 = R0.accKNN(K);  rsrpKNN13 = R0.rsrpKNN(K);  rsrpOpt = R0.summary.rsrp_optimal_dB;

    % ---- representative blockage level: full Top-K curves from the SAME masks ----
    iR = find(blevels==opts.repLevel,1);
    accFc=zeros(nrep,NB); accBc=zeros(nrep,NB); accKNNc=[];
    for r = 1:nrep
        Xrep = Xstore{iR}(:,:,r);
        Rf = evalTopK(predict(netF, Xrep, D.posTest), D);
        Rb = evalTopK(predict(netB, Xrep), D);
        accFc(r,:)=Rf.accNeural; accBc(r,:)=Rb.accNeural;
        if isempty(accKNNc), accKNNc=Rf.accKNN; end
    end
    rep.Kx=1:NB; rep.Fm=mean(accFc,1); rep.Fs=std(accFc,0,1);
    rep.Bm=mean(accBc,1); rep.Bs=std(accBc,0,1); rep.KNN=accKNNc; rep.level=opts.repLevel;

    % ---- figures ----
    makeGatedFigures(blevels, accB, accB_std, accF, accF_std, accKNN13, ...
                     rsB, rsB_std, rsF, rsF_std, rsrpKNN13, rsrpOpt, gateMean, rep, K, figDir);

    % ---- metrics ----
    R = struct('blevels',blevels,'K',K,'nrep',nrep,'repLevel',opts.repLevel, ...
        'accRSRP',accB,'accRSRP_std',accB_std,'accFusion',accF,'accFusion_std',accF_std, ...
        'rsrpRSRP',rsB,'rsrpRSRP_std',rsB_std,'rsrpFusion',rsF,'rsrpFusion_std',rsF_std, ...
        'gateMean',gateMean,'accKNN13',accKNN13,'rsrpKNN13',rsrpKNN13,'rsrpOpt',rsrpOpt);
    if haveArc
        R.accConcat=accC; R.accConcat_std=accC_std; R.accResidual=accR; R.accResidual_std=accR_std;
    end
    R.acc_clean_RSRP = accB(1); R.acc_clean_Fusion = accF(1);
    save(fullfile(metDir,'novel_results.mat'),'R');
    fid=fopen(fullfile(metDir,'novel_metrics.json'),'w'); fwrite(fid, jsonencode(R, PrettyPrint=true)); fclose(fid);

    % ---- console: LaTeX rows for BOTH tables (shared draws => consistent) ----
    iiiC = ismember(blevels,[0 2 4 6 8 10 12]);   % Table III columns
    ivC  = ismember(blevels,[0 6 8 10 12]);        % Table IV columns
    fprintf('\n===== SHARED-DRAW EVAL (acc@K=%d, mean+/-std over %d draws) =====\n', K, nrep);
    fprintf('Table III (cols 0 2 4 6 8 10 12):\n');
    fprintf('  RSRP-only & %s \\\\\n', latexRow(accB(iiiC),accB_std(iiiC)));
    fprintf('  Gated fusion & %s \\\\\n', latexRow(accF(iiiC),accF_std(iiiC)));
    fprintf('  Gain (pts) & %s \\\\\n', gainRow(accF(iiiC)-accB(iiiC)));
    if haveArc
        fprintf('Table IV (cols 0 6 8 10 12):\n');
        fprintf('  RSRP-only      & %s \\\\\n', latexRow(accB(ivC),accB_std(ivC)));
        fprintf('  Concatenation  & %s \\\\\n', latexRow(accC(ivC),accC_std(ivC)));
        fprintf('  Plain residual & %s \\\\\n', latexRow(accR(ivC),accR_std(ivC)));
        fprintf('  Gated residual & %s \\\\\n', latexRow(accF(ivC),accF_std(ivC)));
    end
    fprintf('clean parity gain: %+.1f | mean gate ~ %.2f | KNN acc@K=%d=%.1f%%\n', ...
        accF(1)-accB(1), mean(gateMean), K, accKNN13);
    fprintf('================================================================\n');
end

% ---------------------------------------------------------------------------
function [accAll, rsAll] = evalModelAcc(net, Xstore, D, K, twoInput)
    nL=numel(Xstore); nrep=size(Xstore{1},3);
    accAll=zeros(nL,nrep); rsAll=zeros(nL,nrep);
    for i=1:nL
        for r=1:nrep
            Xn=Xstore{i}(:,:,r);
            if twoInput, p=predict(net,Xn,D.posTest); else, p=predict(net,Xn); end
            [accAll(i,r), rsAll(i,r)] = nnAccRsrp(p, D, K);
        end
    end
end

function [acc, rsrp] = nnAccRsrp(pred, D, K)
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

function s = latexRow(m, sd)
    parts = arrayfun(@(a,b) sprintf('\\ms{%.1f}{%.1f}', a, b), m, sd, 'uni', 0);
    s = strjoin(parts, ' & ');
end
function s = gainRow(g)
    parts = arrayfun(@(x) sprintf('$%+.1f$', x), g, 'uni', 0);
    s = strjoin(parts, ' & ');
end

function makeGatedFigures(bl, accB, accBs, accF, accFs, accKNN, rsB, rsBs, rsF, rsFs, rsKNN, rsOpt, gateMean, rep, K, figDir)
    red=[0.85 0.1 0.1]; blue=[0 0.45 0.74]; org=[0.9 0.6 0];

    f1=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    errorbar(bl, accF, accFs, '-o', LineWidth=2, Color=red,  CapSize=4);
    errorbar(bl, accB, accBs, '-*', LineWidth=2, Color=blue, CapSize=4);
    yline(accKNN, '--', sprintf('KNN (position) %.0f%%',accKNN), Color=org);
    xlabel('Number of blocked input beams (of 14)'); ylabel(sprintf('Top-%d Accuracy (%%)',K));
    title('Robustness to Beam Blockage (mean \pm std, 10 draws)');
    legend('Gated Fusion (RSRP+Pos)','RSRP-only','Location','northeast');
    exportgraphics(f1, fullfile(figDir,'novel_blockage_accuracy.png'), Resolution=200); close(f1);

    f2=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    errorbar(bl, rsF, rsFs, '-o', LineWidth=2, Color=red,  CapSize=4);
    errorbar(bl, rsB, rsBs, '-*', LineWidth=2, Color=blue, CapSize=4);
    yline(rsOpt,'-k',sprintf('Exhaustive %.1f dB',rsOpt));
    xlabel('Number of blocked input beams (of 14)'); ylabel(sprintf('Average RSRP at K=%d (dB)',K));
    title('Average RSRP vs Beam Blockage (mean \pm std)'); legend('Gated Fusion','RSRP-only','Location','southwest');
    exportgraphics(f2, fullfile(figDir,'novel_blockage_rsrp.png'), Resolution=200); close(f2);

    Kx=rep.Kx;
    f3=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    fill([Kx fliplr(Kx)], [rep.Fm+rep.Fs fliplr(rep.Fm-rep.Fs)], red,  FaceAlpha=0.15, EdgeColor='none');
    fill([Kx fliplr(Kx)], [rep.Bm+rep.Bs fliplr(rep.Bm-rep.Bs)], blue, FaceAlpha=0.15, EdgeColor='none');
    h1=plot(Kx, rep.Fm, '-o', LineWidth=2, MarkerIndices=1:3:70, Color=red);
    h2=plot(Kx, rep.Bm, '-*', LineWidth=2, MarkerIndices=1:3:70, Color=blue);
    h3=plot(Kx, rep.KNN,'--s', LineWidth=1.3, MarkerIndices=1:4:70, Color=org);
    yline(90,':','90%');
    xlabel('K'); ylabel('Top-K Accuracy (%)'); xlim([1 70]); ylim([0 100]);
    title(sprintf('Top-K Accuracy at %d/14 Beams Blocked (mean \\pm std)', rep.level));
    legend([h1 h2 h3],'Gated Fusion','RSRP-only','KNN (position)','Location','southeast');
    exportgraphics(f3, fullfile(figDir,'novel_blockage_topk.png'), Resolution=200); close(f3);

    f4=figure('Visible','off','Position',[100 100 720 460]); grid on; hold on;
    plot(bl, gateMean, '-d', LineWidth=2, Color=[0.2 0.5 0.2]);
    ylim([0 1]);
    xlabel('Number of blocked input beams (of 14)'); ylabel('Mean learned gate value');
    title(sprintf('Learned Gate is Nearly Constant (\\approx%.2f)', mean(gateMean)));
    text(0.5, mean(gateMean)+0.06, sprintf('range %.3f-%.3f over all blockage levels', min(gateMean), max(gateMean)));
    exportgraphics(f4, fullfile(figDir,'novel_gate_behavior.png'), Resolution=200); close(f4);
    fprintf('Saved 4 figures to %s\n', figDir);
end
