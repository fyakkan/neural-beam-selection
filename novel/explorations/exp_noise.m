base = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/baseline';
nov  = '/Users/furkanyakkan/Documents/Next Generation Mobile Cominication Final/final/novel';
addpath(base, fullfile(base,'helpers'), nov);
D = loadBeamData(fullfile(base,'data'));
B = load(fullfile(base,'nnBS_regNet.mat'));  netB = B.net;       % RSRP-only (clean-trained)
F = load(fullfile(nov,'nnBS_fusionNet.mat')); netF = F.net;       % RSRP+pos (clean-trained)

NB=D.NumBeamPairs; Nte=D.Ntest; trueOpt=D.optTest(:);
accAtK = @(pred,k) 100*mean(arrayfun(@(n) any(maxk_idx(pred(n,:),k)==trueOpt(n)), 1:Nte));

Xte = D.XtestNN;            % Nte x 14 floored dB
fprintf('Test-time RSRP noise robustness (nets trained on CLEAN RSRP):\n');
fprintf('%6s | %-22s | %-22s\n','sigma','RSRP-only acc@K (8/13/20)','Fusion acc@K (8/13/20)');
rng(7);
for sigma = [0 2 4 6 8 10]
    accs = zeros(2,3);
    nrep = 5;                                  % average over noise draws
    for r=1:nrep
        Xn = Xte + sigma*randn(size(Xte));
        pB = predict(netB, Xn);
        pF = predict(netF, Xn, D.posTest);
        kk=[8 13 20];
        for j=1:3
            accs(1,j)=accs(1,j)+accAtK(pB,kk(j))/nrep;
            accs(2,j)=accs(2,j)+accAtK(pF,kk(j))/nrep;
        end
    end
    fprintf('%6.0f | %5.1f %5.1f %5.1f          | %5.1f %5.1f %5.1f   (gain@13 %+.1f)\n', ...
        sigma, accs(1,:), accs(2,:), accs(2,2)-accs(1,2));
end
disp('EXP_NOISE_DONE');

function idx = maxk_idx(v,k); [~,idx]=maxk(v,k); end
