function x = scale_to_01(x)
    % Scale all elements in the array x, so that it's values lies in i
    % the range [0, 1].
    %
    x = (x - min(x(:)))/(max(x(:)) - min(x(:)));
end

