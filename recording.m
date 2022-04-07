function [spikes,noise,toa,peaks] = recording(N,fs,cells,lambda,spike_base,noise_temp)
%fs = 24414;cells = 2;lambda = 20;
% for i=1:size(noise_base,2)
%     noise_base(:,i)=noise_base(:,i)/max(noise_base(:,i));
% end
%N=1e5;
D=64; %duration of noise base
T=N/fs;


ToA=cell(cells,1);
for i = 1:cells % toa follows expotiontial distrib.-> can be upgraded to nonhomogenous poission distrib.
    toa=cumsum(exprnd(1/lambda,round(T*(1+lambda)),1));
    ToA{i}=ceil(toa(toa<=T)*fs);
end

spikes=0;
% noise_energy=0;
numArrival=0;
toa=[];
peaks = [];
for i =1:cells % noise for each cell
    numArrival=numArrival+length(ToA{i});
    toa_temp=ToA{i};
    toa=[toa;toa_temp+19];
    noise_sel=spike_base(:,randi(1000,1,length(toa_temp)));
    peaks = [peaks,max(noise_sel)];
    for j =1:length(toa_temp)
        noise_mat=zeros(N,1);
        if toa_temp(j)+D<=N
            noise_mat(toa_temp(j):toa_temp(j)+D-1)=noise_sel(:,j);
%             noise_energy=noise_energy+sum(noise_sel(:,j).^2);
        end
        spikes=spikes+noise_mat;
    end
end
peaks_mean = mean(peaks);
spikes = spikes / peaks_mean;
peaks = peaks/peaks_mean;
rnd = round((length(noise_temp) - N) *rand(1));
noise = double(noise_temp(rnd+1:rnd+N));
noise = noise/std(noise);

