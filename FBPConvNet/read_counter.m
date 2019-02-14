function count = read_counter(path)

    fID = fopen(path, 'r');

    count = fscanf(fID, '%f');
    fclose(fID);

    fID = fopen(path, 'w');
    fprintf(fID, '%d\n', count + 1);
    fclose(fID);

end 

