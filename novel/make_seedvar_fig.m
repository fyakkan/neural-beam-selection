function make_seedvar_fig()
%make_seedvar_fig  Regenerate the blockage-accuracy figure with error bars that
%   reflect ACROSS-SEED variance (from exp_seed_variance.m / seed_variance.mat),
%   matching the seed-variance Table III. Writes the PNG into BOTH the repo
%   results/figures and the paper/figures directory the paper includes from.
%
%   Run after exp_seed_variance: matlab -batch "make_seedvar_fig"

    here   = fileparts(mfilename('fullpath'));
    repo   = fileparts(here);
    metDir = fullfile(repo,'results','metrics');
    figDir = fullfile(repo,'results','figures');
    paperFig = fullfile(repo,'paper','figures');

    S = load(fullfile(metDir,'seed_variance.mat')); R = S.R;
    bl = R.blevels;
    accB = R.accB_mean; accBs = R.accB_sd;
    accF = R.accF_mean; accFs = R.accF_sd;
    accKNN = 37.8;   % position-only KNN at K=13 (blockage-immune, from baseline eval)
    nS = R.nseeds;

    red=[0.85 0.1 0.1]; blue=[0 0.45 0.74]; org=[0.9 0.6 0];
    f1=figure('Visible','off','Position',[100 100 720 500]); hold on; grid on;
    errorbar(bl, accF, accFs, '-o', LineWidth=2, Color=red,  CapSize=4);
    errorbar(bl, accB, accBs, '-*', LineWidth=2, Color=blue, CapSize=4);
    yline(accKNN, '--', sprintf('KNN (position) %.0f%%',accKNN), Color=org);
    xlabel('Number of blocked input beams (of 14)'); ylabel('Top-13 Accuracy (%)');
    title(sprintf('Robustness to Beam Blockage (mean \\pm std across %d seeds)', nS));
    legend('Gated Fusion (RSRP+Pos)','RSRP-only','Location','northeast');
    exportgraphics(f1, fullfile(figDir,'novel_blockage_accuracy.png'), Resolution=200);
    exportgraphics(f1, fullfile(paperFig,'novel_blockage_accuracy.png'), Resolution=200);
    close(f1);
    fprintf('Wrote novel_blockage_accuracy.png (across-seed error bars) to:\n  %s\n  %s\n', figDir, paperFig);
    disp('MAKE_SEEDVAR_FIG_DONE');
end
