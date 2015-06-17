function ShowModel( mu, pi, n_rows, n_cols, ind )
    
    mu = reshape(mu, 20, 20, length(pi));
    k = 1;
    for r = 1 : n_rows
        for c = 1 : n_cols
            if (k <= length(ind))
                subplot(n_rows, n_cols, k), ...
                    imshow(mu(:,:,ind(k)), [0 1]), ...
                    xlabel(num2str(pi(ind(k))*100, '%.4f')), ...
                    set(gca, 'xtick', [], 'ytick', []);
                    % title(num2str(k, 'cluster %d'));
                k = k + 1;
            else
                return
            end
        end
    end
end

