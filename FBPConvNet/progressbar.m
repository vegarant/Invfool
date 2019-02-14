function progressbar(i, N, bar_length);
    % progressbar() creates a progressbar within in command promptpt. It is
    % intended for use in for-loops looping over the indices 1:N.
    % 
    % i:          iteration parameter in the loop
    % N:          Size of the loop 
    % bar_length: length of the progress bar (default is 70)
    %
    % USAGE:
    % >> N = 10000
    % >> 
    % >> for i = 1:N
    % >>     progressbar(i, N);
    % >>     pause(0.001);
    % >> end
    %
    if (nargin < 3)
        bar_length = 72;
    end

    num_prev = round((i-1)*bar_length/N);
    num = round(i*bar_length/N);

    if (i ~= 1)
        if (num_prev ~= num) % Draw a new line
            A = zeros([num,1], 'uint8');
            A(:) = '#';
            space = zeros(bar_length-num, 1, 'uint8');
            space(:) = ' ';
            fprintf('\r[%s%s]', A, space);
        end
    else
        space = zeros(bar_length, 1, 'uint8');
        space(:) = ' ';
        fprintf('\r[%s]', space);
    end

    if (i == N)
        fprintf('\n');
    end

end
