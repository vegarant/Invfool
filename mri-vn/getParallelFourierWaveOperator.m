function A = getParallelFourierWaveOperator(m, mask, coil_sens, nres, vm)

    dwtmode('per', 'nodisp');

    vec = @(x) x(:);

    if length(m) == 3
        A.m         = m;
        A.mask      = mask;
        A.coil_sens = coil_sens;
        A.nres      = nres;
        A.wname = sprintf('db%d', vm);
        [c,S] = wavedec2(ones([m(1), m(2)]), A.nres, A.wname);

        A.S = S;
        % create Mask as operator, mask is centered in k-space
        % mask = opMask(prod(m),mask);

        % sampling in k-space
        A.times = @(x) forwardOp(x,  A.m, A.mask, A.coil_sens, A.wname, A.S);
        A.adj   = @(x) backwardOp(x, A.m, A.mask, A.coil_sens, A.wname, A.nres);
        %A.times = @(x) vec((A.mask).*(fftshift(fftshift(fft2(reshape(x,m)),1),2)))./scale_fact;
        %A.adj   = @(x) vec(sum(ifft2(ifftshift(ifftshift(reshape((A.mask).*(reshape(x,m)),m),1),2)), 3).*scale_fact));
    end


end

function y = forwardOp(x, m, mask, coil_sens, wname, S)
    % x - [heigth, width]
    c = waverec2(real(x), S, wname)+ 1j*waverec2(imag(x),S, wname);
    scale_factor = sqrt(m(1)*m(2));
    x = repmat(c, 1,1,m(3));

    coil_im = coil_sens.*x;
    fft2_coil_im = fftshift(fftshift(fft2(coil_im), 1), 2)/scale_factor;
    y = mask.* fft2_coil_im;
    y = y(:);

end

function x = backwardOp(y, m, mask, coil_sens, wname, nres)
    % x         - [heigth, width, channels]
    % mask      - [heigth, width, channels]
    % coil_sens - [heigth, width, channels]

    scale_factor = sqrt(m(1)*m(2));
    y = reshape(y, m);
    y = mask.*y;

    ifft2_im = ifft2(ifftshift(ifftshift(y, 2), 1))*scale_factor;
    coil_im_adj = conj(coil_sens).*ifft2_im;
    x = sum(coil_im_adj, 3);

    x = wavedec2(real(x), nres, wname) + 1j*wavedec2(imag(x), nres, wname);
    x = x(:);

end




