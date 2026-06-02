function X = applyImpairment(X, sigma, nBlock, q, floorVal)
%applyImpairment  Apply realistic RSRP measurement impairments to NN inputs.
%
%   X = applyImpairment(X, SIGMA, NBLOCK, Q, FLOORVAL) takes sampled RSRP
%   measurements X (numObs x numSampled, floored dB) and returns an impaired
%   copy modelling practical sensing:
%       - blockage : NBLOCK randomly-chosen input beams per observation are
%                    lost (set to FLOORVAL), modelling blocked/failed sweeps.
%       - quantization : RSRP rounded to Q-dB steps (Q=0 disables).
%       - noise    : additive Gaussian measurement noise, std SIGMA dB.
%
%   Vectorised over observations; blocked beams are drawn independently per row.
%   Used both as a test-time impairment and (with randomised levels) as a
%   training-time augmentation for impairment-aware models.

    arguments
        X double
        sigma    (1,1) double = 0
        nBlock   (1,1) double = 0
        q        (1,1) double = 0
        floorVal (1,1) double = -120
    end

    [N, M] = size(X);

    if nBlock > 0
        nBlock = min(nBlock, M);
        for n = 1:N
            bi = randperm(M, nBlock);
            X(n, bi) = floorVal;
        end
    end

    if q > 0
        X = round(X./q).*q;
    end

    if sigma > 0
        X = X + sigma*randn(N, M);
    end
end
