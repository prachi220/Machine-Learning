function normX = normalise( X )
    u = 0 ;
    temp =0;
    len = size(X);
    for i = 1:size(X)
        u = u + X(i);
    end
    u = u / length(X);
    for i = 1:size(X)
        temp =  temp + (X(i) - u )^2;
    end

    temp = temp/length(X);
    variance = sqrt(temp);
    normX  = (X - u)/ variance;

end

