function y = simple_spgl1_op(x, mode, A)
    if (mode == 1)
        y = A.times(x);
    else
        y = A.adj(x);
    end
end
