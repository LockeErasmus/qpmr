clear all;
close all;

%region = [-2, 7, 0, 9]
%region = [-10, 7, 0, 100]
region = [-10, 7, 0, 200]
delays = [0.0, 1.3, 3.5, 4.3]
python_coefs = [[20.1, 0, 0.2, 1.5],
                [0, -2.1, 0, 1],
                [0, 3.2, 0, 0],
                [1.4, 0, 0, 0]]

coefs = fliplr(python_coefs)

N = 1001; % one more, dont count the first run, that has huge overhead in MATLAB
time_vector = zeros(1,N);

for i=1:N
    tstart = tic;
    QPmR(region, coefs, delays);
    telapsed = toc(tstart);
    time_vector(i) = telapsed;
end

mean(time_vector(2:end))
std(time_vector(2:end))
N-1
region