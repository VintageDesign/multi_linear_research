N = 100;
StartN = 2;
NumTests = 10000;
averages = [];
for n = StartN:N
    A = rand(n, n);
    curr_times = [];
    for i = 1:NumTests
        tic;
        tmp = inv(A);
        curr_time = toc;
        curr_times{i} = curr_time;
    end
    averages{n-StartN+1} = mean(cell2mat(curr_times));
end