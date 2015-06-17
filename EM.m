function EM()

    clc; clear; tic;
    addpath('./utils');
    [train, test] = DataPrep('./data');
    toc;

    K = 40;
    N = 50000;
    D = 400;
    
    pi_ = zeros(1, K);
    for j = 1:K
        pi_(j) = 1 / K;
    end
    
    mu = zeros(K, D);
    for w = 1:K
        
        normalizationFactor = 0;
        for g = 1:D
            mu(w, g) = rand();
            normalizationFactor = normalizationFactor + mu(w, g);
        end
        
        for g = 1:D
            mu(w, g) = mu(w, g) / normalizationFactor;
        end
    end
    
    x = train.images.';
    
    z = zeros(N, K);
    
    for r = 1:1
        ExpectationStep();
        disp(r);
        MaximizationStep();
    end
    
    ShowModel(mu, pi_, 5, 8, 1:40);
    
    function ExpectationStep()
        
        for n= 1:N
            normalizationFactor = 0;
        
            for k = 1:K
                z(n, k) = ExpectationSubstep(n, k);
                normalizationFactor = normalizationFactor + z(n, k);
            end
        
            for k = 1:K
                if normalizationFactor > 0
                    z(n, k) = z(n, k) / normalizationFactor;
                else
                    z(n, k) = 1 / K;
                end 
                
            end 
            
        end
        
    end

    function MaximizationStep()

        for k = 1:K
            pi_(k) = (Nm(k) / N);
        end
    
        for k = 1:K
            averageX_k = AverageX(k);
        
            for i = 1:D
                mu(k, i) = averageX_k(i);
            end
            
        end
        
    end
 
    function z_nk = ExpectationSubstep(n, k)

        z_nk = pi_(k);
        for i = 1:D
            z_nk = z_nk * (mu(k, i) .^  x(n, i)) * ((1 - mu(k, i)) .^ (1 - x(n, i)));
        end
    
    end

    function result = AverageX(m)

        result = zeros(1, D);
        for i = 1:D
            for n = 1:N
                result(i) = result(i) + z(n, m) * x(n, i);
            end
        end
    
        currentNm = Nm(m);
        for i = 1:D
            result(i) = result(i) / currentNm;
        end
    
    end

    function result = Nm(m)
        result = 0;
        for n = 1:N
            result = result + z(n, m);
        end
    end

end
