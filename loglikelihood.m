function loglikelihood(x, mu, pi_, z)
    N = length(x);
    disp(N);
    K = 40;
    D = 400;
    result = 0;
    for n = 1:N
        for k = 1:K
            
            temp = 0;
            for i = 1:D
                temp = temp + x(n, i) * log(mu(k, i)) + (1 - x(n, i)) * log(1 - mu(k, i));
            end
            result = result + z(i, k) * (log(pi_(k)) + temp);
        end
    end
    disp(result)
end