function A = getParallelFourierOperator(m, mask, coil_sens)

    vec = @(x) x(:);

    if length(m) == 3
        A.m         = m;
        A.mask      = mask;
        A.coil_sens = coil_sens;

        % sampling in k-space
        A.times = @(x) forwardOp(x,  A.m, A.mask, A.coil_sens);
        A.adj   = @(x) backwardOp(x, A.m, A.mask, A.coil_sens);

    end

end

function y = forwardOp(x, m, mask, coil_sens)
    % x - [heigth, width]

    scale_factor = sqrt(m(1)*m(2));
    x = reshape(x, [m(1), m(2)]);
    x = repmat(x, 1,1,m(3));
    coil_im = coil_sens.*x;
    fft2_coil_im = fftshift(fftshift(fft2(coil_im),1),2)/scale_factor;
    y = mask.* fft2_coil_im;
    y = y(:);

end

function x = backwardOp(y, m, mask, coil_sens)
    % x         - [heigth, width, channels]
    % mask      - [heigth, width, channels]
    % coil_sens - [heigth, width, channels]

    scale_factor = sqrt(m(1)*m(2));
    y = reshape(y, m);
    y = mask.*y;

    ifft2_im = ifft2(ifftshift(ifftshift(y,2),1))*scale_factor;
    coil_im_adj = conj(coil_sens).*ifft2_im;
    x = sum(coil_im_adj, 3);
    x = x(:);

end

