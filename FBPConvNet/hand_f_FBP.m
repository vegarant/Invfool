function y = hand_f_FBP(net, x0)
% Apply the network to the filtered backprojection x0.
%
% INPUT:
% net - network struct
% x0  - Filtered backprojection of sinogram image
%
% OUTPUT:
% y - Artifact free image, after sending x0 through the network.


    res1 = vl_simplenn_fbpconvnet(net,gpuArray((single(x0))), single(1));
    y = gather(res1(end-1).x + x0);

end
