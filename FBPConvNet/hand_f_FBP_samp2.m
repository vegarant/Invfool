function y = hand_f_FBP_samp2(net, x0, theta)
    % Samples the image `x0`, at the angles `theta` and perform reconstruction
    % using the FBPConvNet `net`.
    %
    % INPUT:
    % net   - FBPConvNet structure
    % x0    - Clear image
    % theta - Angles in the range [0,180).
    %
    % OUTPUT
    % y - The FBPConvNet reconstruction
    %
    x1 = iradon( radon(x0, theta)  , theta, 'linear', 'Ram-Lak', 1, 512);
    res1 = vl_simplenn_fbpconvnet(net,gpuArray((single(x1))), single(1));
    y = gather(res1(end-1).x + x1);

end

