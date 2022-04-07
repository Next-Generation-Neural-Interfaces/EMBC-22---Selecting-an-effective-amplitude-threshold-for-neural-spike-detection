%% noise mean
load noise_base.mat
load realDataWithLFP_1.mat
load spike_location_1.mat
noise_temp = extractMean(data,16);
for j = 1:length(spike_location)
    noise_temp(spike_location(j)-10:spike_location(j)+10) = 1000; %removel all spikes
end
noise_temp = noise_temp(noise_temp ~= 1000);
N=1e6;fs = 24414;cells = 1;lambda = 20;
D = 30;


[spike,noise,toa,peaks] = recording(N,fs,cells,lambda,noise_base,noise_temp);

spike_loc = toa;
COLORS = get(groot, 'factoryAxesColorOrder');
COLORS = COLORS - 0.2;
COLORS(COLORS < 0) = 0;

% plot(data_noised);hold on;
% scatter(spike_loc+1,data_noised(spike_loc+1));hold off;

SNRstart = 0;
SNRend = 10;
SNR = 5:2:40;
multiplier = 1:50;
before = 3;
after = 3;
thr = zeros(length(SNR),length(multiplier));
Acc = zeros(length(SNR),length(multiplier));
Sens = zeros(length(SNR),length(multiplier));
FDR = zeros(length(SNR),length(multiplier));
for i = 1:length(SNR)
    data_noised  = spike + noise/SNR(i);
    noise_mean = mean(abs(noise/SNR(i)));
%     noise_mean = std(noise'/SNR(i));
%      data_noised  = spike*SNR(i) + noise';
     
    
    parfor j = 1:length(multiplier)
        thr(i,j)  = multiplier(j)* noise_mean;
        spikes = find(data_noised>thr(i,j));
        ispike_detected = [];
        iinterval = [];
        add_factor = 0;
        while(length(spikes)>1)
            cur_range = spikes((spikes <= (spikes(1)+D)) & (spikes >= spikes(1)));
            [~,idx] = max(data_noised(cur_range+add_factor));
            ispike_detected = [ispike_detected, cur_range(idx)+add_factor];
            iinterval = [iinterval;[spikes(1)+add_factor-10,spikes(1)+add_factor+50]];
            if(size(ispike_detected,2) ~= size(iinterval))
                a = 0;
            end
            sub_factor =  spikes(1) + D;
            add_factor = add_factor + spikes(1)+D;
            spikes = spikes-sub_factor;
            spikes = spikes(spikes>0);
        end
        iinterval(iinterval<1) = 1;
        iinterval(iinterval>length(data_noised)) = length(data_noised);

%         if(i == 15)
%             figure(11)
%             hold on
%             plot(data_noised);
%             plot(thr(i,j)*ones(size(data_noised)));
% %             scatter(spike_loc,data_noised(spike_loc),'.');
% %             scatter(ispike_detected,data_noised(ispike_detected));hold off
%         end
%         if(j == 15)
%             figure
%             plot(data_noised);hold on
%             plot(thr(i,j)*ones(size(data_noised)));
%             scatter(spike_loc,data_noised(spike_loc),'.');
%             scatter(ispike_detected,data_noised(ispike_detected));hold off
%         end
        [FP,FN,TP]=locationCompare(spike_loc,iinterval,ispike_detected);
        Sens(i,j)= length(TP)/(length(TP)+length(FN)); % found is correct
        FDR(i,j) = length(FP)/(length(FP)+length(TP)); % not find
        Acc(i,j) = length(TP)/(length(TP)+length(FN)+length(FP));
    end
end   
figure(10)
plot(multiplier,Acc'); hold on
for i = 1:size(Acc,1)
    [m,idx] = max(Acc(i,:));
%     plot(multiplier(idx-2:idx+2), Acc(i,idx-2:idx+2),'k.','LineWidth',2)
    scatter(multiplier(idx),Acc(i,idx),'k*')
end
hold off
max_idx = -1000;
min_idx = 1000;
for i = 1:size(Acc,1)
    [m,idx] = max(Acc(i,:));
    if(m>0.8)
        if(idx > max_idx)
            max_idx = idx;
        end
        if(idx<min_idx)
            min_idx = idx;
        end
    end
end
shift = multiplier(max_idx) - multiplier(min_idx);

span = 0;
count = 0;
for i = 1:size(Acc,1)
    mul_accept = multiplier(Acc(i,:)>0.8);
    if(~isempty(mul_accept))
        count = count + 1;
        span = span + mul_accept(end) - mul_accept(1);
    end
end
span = span/count;
span7 = 0;
count = 0;
for i = 1:size(Acc,1)
    mul_accept = multiplier(Acc(i,:)>0.7);
    if(~isempty(mul_accept))
        count = count + 1;
        span7 = span7 + mul_accept(end) - mul_accept(1);
    end
end
span7 = span7/count;
s = span/span7;

factor = shift/span;

xlabel('$\alpha$')
ylabel('Accuracy')
title(['SR = ', num2str(s,2),' SSR = ',num2str(factor,2)])


% figure(2)
% plot(multiplier,FDR'); 
% figure(3)
% plot(multiplier,Sens'); 
%% noise mean & std
load noise_base.mat
load realDataWithLFP_1.mat
load spike_location_1.mat
noise_temp = extractMean(data,16);
for j = 1:length(spike_location)
    noise_temp(spike_location(j)-10:spike_location(j)+10) = 1000;
end
noise_temp = noise_temp(noise_temp ~= 1000);
N=1e6;fs = 24414;cells = 1;lambda = 20;
D = 30;


% [spike,noise,toa,peaks] = recording(N,fs,cells,lambda,noise_base,noise_temp);




spike_loc = toa;
COLORS = get(groot, 'factoryAxesColorOrder');
COLORS = COLORS - 0.2;
COLORS(COLORS < 0) = 0;

% plot(data_noised);hold on;
% scatter(spike_loc+1,data_noised(spike_loc+1));hold off;

SNRstart = 0;
SNRend = 10;
SNR = 1:4:40;
multiplier = 1:20;
    
before = 3;
after = 3;
thr = zeros(length(SNR),length(multiplier),length(multiplier));
Acc = zeros(length(SNR),length(multiplier),length(multiplier));
Sens = zeros(length(SNR),length(multiplier),length(multiplier));
FDR = zeros(length(SNR),length(multiplier),length(multiplier));
for i = 1:length(SNR)
    data_noised  = spike + noise/SNR(i);
%     %data_noised =  addNoise(data,SNR(i));  
    noise_mean = mean(abs(noise/SNR(i)));
    noise_std = std(noise/SNR(i));
%     spike_mean = 1;
%      noise_mean = mean(abs(noise));
%     noise_mean = std(noise'/SNR(i));
%      data_noised  = spike*SNR(i) + noise';
     
    
       
    for j = 1:length(multiplier)
        parfor k = 1:length(multiplier)
            thr(i,j,k)  = multiplier(j)*noise_mean + multiplier(k)* noise_std;
            spikes = find(data_noised>thr(i,j,k));
            ispike_detected = [];
            iinterval = [];
            add_factor = 0;
            while(length(spikes)>1)
                cur_range = spikes((spikes <= (spikes(1)+D)) & (spikes >= spikes(1)));
                [~,idx] = max(data_noised(cur_range+add_factor));
                ispike_detected = [ispike_detected, cur_range(idx)+add_factor];
                iinterval = [iinterval;[spikes(1)+add_factor-10,spikes(1)+add_factor+50]];
                if(size(ispike_detected,2) ~= size(iinterval))
                    a = 0;
                end
                sub_factor =  spikes(1) + D;
                add_factor = add_factor + spikes(1)+D;
                spikes = spikes-sub_factor;
                spikes = spikes(spikes>0);
            end
            iinterval(iinterval<1) = 1;
            iinterval(iinterval>length(data_noised)) = length(data_noised);
    %         if(j == 15)
    %             figure
    %             plot(data_noised);hold on
    %             plot(thr(i,j)*ones(size(data_noised)));
    %             scatter(spike_loc,data_noised(spike_loc),'.');
    %             scatter(ispike_detected,data_noised(ispike_detected));hold off
    %         end
            [FP,FN,TP]=locationCompare(spike_loc,iinterval,ispike_detected);
            Sens(i,j,k)= length(TP)/(length(TP)+length(FN)); % found is correct
            FDR(i,j,k) = length(FP)/(length(FP)+length(TP)); % not find
            Acc(i,j,k) = length(TP)/(length(TP)+length(FN)+length(FP));
        end
    end
end   
% figure(1)
% plot(multiplier,Acc'); 
% figure(2)
% plot(multiplier,FDR'); 
figure(7)
for i = 1:length(SNR)
    toPlot = permute(Acc(i,:,:),[2,3,1]);
%     surf(multiplier,multiplier,toPlot,'EdgeColor', COLORS(mod(i,7)+1,:));hold on
    x = [];y = [];z = [];
    for j = 1:size(toPlot,1)
        x = [x ,multiplier(j)];
        [~,idx] = max(toPlot(j,:));
        y = [y, multiplier(idx)];
        z = [z, toPlot(j,idx)];
    end
    plot(x,y,'LineWidth',2);hold on
%     pause(1)
    
end
    xlabel('$\alpha$')
    ylabel('$\beta$')
    zlabel('Accuracy')
hold off
legend('5','9','13','17','21','25','19','33','37')
% figure(3)
% plot(multiplier,Sens'); 
%% spikes
load noise_base.mat
load realDataWithLFP_1.mat
load spike_location_1.mat
noise_temp = extractMean(data,16);
for j = 1:length(spike_location)
    noise_temp(spike_location(j)-10:spike_location(j)+10) = 1000;
end
noise_temp = noise_temp(noise_temp ~= 1000);
N=1e6;fs = 24414;cells = 1;lambda = 20;
D = 30;


% [spike,noise,toa,peaks] = recording(N,fs,cells,lambda,noise_base,noise_temp);

spike_loc = toa;
COLORS = get(groot, 'factoryAxesColorOrder');
COLORS = COLORS - 0.2;
COLORS(COLORS < 0) = 0;

% plot(data_noised);hold on;
% scatter(spike_loc+1,data_noised(spike_loc+1));hold off;

SNRstart = 0;
SNRend = 10;
SNR = 5:2:40;
% multiplier = 0.025:0.025:1;
multiplier = 0.03:0.025:1.25;

before = 3;
after = 3;
thr = zeros(length(SNR),length(multiplier));
Acc = zeros(length(SNR),length(multiplier));
Sens = zeros(length(SNR),length(multiplier));
FDR = zeros(length(SNR),length(multiplier));
for i = 1:length(SNR)
%     data_noised  = spike + noise'/SNR(i);
    %data_noised =  addNoise(data,SNR(i));
     noise_mean = mean(abs(noise));
%     noise_mean = std(noise'/SNR(i));
     data_noised  = spike*SNR(i) + noise;
     
    
    parfor j = 1:length(multiplier)
        thr(i,j)  = multiplier(j)*SNR(i);%*noise_mean;
        spikes = find(data_noised>thr(i,j));         
        ispike_detected = [];
        iinterval = [];
        add_factor = 0;
        while(length(spikes)>1)
            cur_range = spikes((spikes <= (spikes(1)+D)) & (spikes >= spikes(1)));
            [~,idx] = max(data_noised(cur_range+add_factor));
            ispike_detected = [ispike_detected, cur_range(idx)+add_factor];
            iinterval = [iinterval;[spikes(1)+add_factor-10,spikes(1)+add_factor+50]];
            if(size(ispike_detected,2) ~= size(iinterval))
                a = 0;
            end
            sub_factor =  spikes(1) + D;
            add_factor = add_factor + spikes(1)+D;
            spikes = spikes-sub_factor;
            spikes = spikes(spikes>0);
        end
        iinterval(iinterval<1) = 1;
        iinterval(iinterval>length(data_noised)) = length(data_noised);
%         if(i == 15)
%             figure(12)
%             hold on
%             plot(data_noised);
%             plot(thr(i,j)*ones(size(data_noised)));
% %             scatter(spike_loc,data_noised(spike_loc),'.');
% %             scatter(ispike_detected,data_noised(ispike_detected));hold off
%         end
%         if(j == 15)
%             figure
%             plot(data_noised);hold on
%             plot(thr(i,j)*ones(size(data_noised)));
%             scatter(spike_loc,data_noised(spike_loc),'.');
%             scatter(ispike_detected,data_noised(ispike_detected));hold off
%         end
        [FP,FN,TP]=locationCompare(spike_loc,iinterval,ispike_detected);
        Sens(i,j)= length(TP)/(length(TP)+length(FN)); % found is correct
        FDR(i,j) = length(FP)/(length(FP)+length(TP)); % not find
        Acc(i,j) = length(TP)/(length(TP)+length(FN)+length(FP));
    end
end   
figure(11)
plot(multiplier,Acc'); hold on
for i = 1:size(Acc,1)
    [m,idx] = max(Acc(i,:));
%     plot(multiplier(idx-2:idx+2), Acc(i,idx-2:idx+2),'k','LineWidth',2)
    scatter(multiplier(idx),Acc(i,idx),'k*')
end
hold off

max_idx = -1000;
min_idx = 1000;
for i = 1:size(Acc,1)
    [m,idx] = max(Acc(i,:));
    if(m>0.8)
        if(idx > max_idx)
            max_idx = idx;
        end
        if(idx<min_idx)
            min_idx = idx;
        end
    end
end
shift = multiplier(max_idx) - multiplier(min_idx);

span = 0;
count = 0;
for i = 1:size(Acc,1)
    mul_accept = multiplier(Acc(i,:)>0.8);
    if(~isempty(mul_accept))
        count = count + 1;
        span = span + mul_accept(end) - mul_accept(1);
    end
end
span = span/count;
span7 = 0;
count = 0;
for i = 1:size(Acc,1)
    mul_accept = multiplier(Acc(i,:)>0.7);
    if(~isempty(mul_accept))
        count = count + 1;
        span7 = span7 + mul_accept(end) - mul_accept(1);
    end
end
span7 = span7/count;
s = span/span7;

factor = shift/span;

xlabel('$\alpha$')
ylabel('Accuracy')
title(['SR = ', num2str(s,2),' SSR = ',num2str(factor,2)])
% figure(2)
% plot(multiplier,FDR'); 
% figure(3)
% plot(multiplier,Sens'); 
%%
plot(spike*10 + noise');hold on;
plot(spike*10);hold off;
%% noise - spikes
load noise_base.mat
load realDataWithLFP_1.mat
load spike_location_1.mat
noise_temp = extractMean(data,16);
for j = 1:length(spike_location)
    noise_temp(spike_location(j)-10:spike_location(j)+10) = 1000;
end
noise_temp = noise_temp(noise_temp ~= 1000);
N=1e5;fs = 24414;cells = 2;lambda = 20;
D = 30;


% [spike,noise,toa,peaks] = recording(N,fs,cells,lambda,noise_base,noise_temp);

spike_loc = toa;
COLORS = get(groot, 'factoryAxesColorOrder');
COLORS = COLORS - 0.2;
COLORS(COLORS < 0) = 0;

% plot(data_noised);hold on;
% scatter(spike_loc+1,data_noised(spike_loc+1));hold off;

SNRstart = 0;
SNRend = 10;
SNR = 5:4:40;
multiplier = 0.1:0.05:2;
    
before = 3;
after = 3;
thr = zeros(length(SNR),length(multiplier),length(multiplier));
Acc = zeros(length(SNR),length(multiplier),length(multiplier));
Sens = zeros(length(SNR),length(multiplier),length(multiplier));
FDR = zeros(length(SNR),length(multiplier),length(multiplier));
for i = 1:length(SNR)
%     data_noised  = spike + noise'/SNR(i);
%     %data_noised =  addNoise(data,SNR(i));  
%     noise_mean = mean(abs(noise'/SNR(i)));
%     noise_std = std(noise'/SNR(i));
%     spike_mean = 1;
     noise_mean = mean(abs(noise));
%     noise_mean = std(noise'/SNR(i));
     data_noised  = spike*SNR(i) + noise;
     
    
       
    for j = 1:length(multiplier)
        parfor k = 1:length(multiplier)
            thr(i,j,k)  = multiplier(j)*SNR(i) + multiplier(k)*5 * noise_mean;
            spikes = find(data_noised>thr(i,j,k));
            ispike_detected = [];
            iinterval = [];
            add_factor = 0;
            while(length(spikes)>1)
                cur_range = spikes((spikes <= (spikes(1)+D)) & (spikes >= spikes(1)));
                [~,idx] = max(data_noised(cur_range+add_factor));
                ispike_detected = [ispike_detected, cur_range(idx)+add_factor];
                iinterval = [iinterval;[spikes(1)+add_factor-10,spikes(1)+add_factor+50]];
                if(size(ispike_detected,2) ~= size(iinterval))
                    a = 0;
                end
                sub_factor =  spikes(1) + D;
                add_factor = add_factor + spikes(1)+D;
                spikes = spikes-sub_factor;
                spikes = spikes(spikes>0);
            end
            iinterval(iinterval<1) = 1;
            iinterval(iinterval>length(data_noised)) = length(data_noised);
    %         if(j == 15)
    %             figure
    %             plot(data_noised);hold on
    %             plot(thr(i,j)*ones(size(data_noised)));
    %             scatter(spike_loc,data_noised(spike_loc),'.');
    %             scatter(ispike_detected,data_noised(ispike_detected));hold off
    %         end
            [FP,FN,TP]=locationCompare(spike_loc,iinterval,ispike_detected);
            Sens(i,j,k)= length(TP)/(length(TP)+length(FN)); % found is correct
            FDR(i,j,k) = length(FP)/(length(FP)+length(TP)); % not find
            Acc(i,j,k) = length(TP)/(length(TP)+length(FN)+length(FP));
            
        end
    end
end   
% figure(1)
% plot(multiplier,Acc'); 
% figure(2)
% plot(multiplier,FDR'); 
figure(7)
for i = 1:length(SNR)
    toPlot = permute(Acc(i,:,:),[2,3,1]);
    surf(multiplier,multiplier,toPlot,'EdgeColor', COLORS(mod(i,7)+1,:));hold on
    x = [];y = [];z = [];
    for j = 1:size(toPlot,1)
        x = [x ,multiplier(j)];
        [~,idx] = max(toPlot(j,:));
        y = [y, multiplier(idx)];
        z = [z, toPlot(j,idx)];
    end
    plot3(y,x,z,'k','LineWidth',2);
%     pause(1)
    
end
    xlabel('$\alpha$')
    ylabel('$\beta$')
    zlabel('Accuracy')
hold off
% legend('5','9','13','17','21','25','19','33','37')
% figure(3)
% plot(multiplier,Sens'); 
%% beta slides
load noise_base.mat
load realDataWithLFP_1.mat
load spike_location_1.mat
noise_temp = extractMean(data,16);

spikes = find(noise_temp>0.7e-4);

for j = 1:length(spikes)
    noise_temp(spikes(j)-10:spikes(j)+10) = 1000;
end
noise_temp = noise_temp(noise_temp ~= 1000);

N=1e5;fs = 24414;cells = 1;lambda = 30;
D = 10;


% [spike,noise,toa,peaks] = recording(N,fs,cells,lambda,noise_base,noise_temp);

spike_loc = toa;
COLORS = get(groot, 'factoryAxesColorOrder');
COLORS = COLORS - 0.2;
COLORS(COLORS < 0) = 0;

% plot(data_noised);hold on;
% scatter(spike_loc+1,data_noised(spike_loc+1));hold off;

SNRstart = 0;
SNRend = 10;
SNR = 5:2:40;
multiplier = -18:0.5:18;
before = 3;
after = 3;
thr = zeros(length(SNR),length(multiplier));
Acc = zeros(length(SNR),length(multiplier));
Sens = zeros(length(SNR),length(multiplier));
FDR = zeros(length(SNR),length(multiplier));
for i = 1:length(SNR)
    i
%     data_noised  = spike + noise'/SNR(i);
    %data_noised =  addNoise(data,SNR(i));
     noise_mean = mean(abs(noise));
%     noise_mean = std(noise'/SNR(i));
     data_noised  = spike*SNR(i) + noise;
     
    
    parfor j = 1:length(multiplier)
        thr(i,j)  =0.6*SNR(i) + multiplier(j) * noise_mean;
        spikes = find(data_noised>thr(i,j));
        if length(spikes) > length(data_noised) *0.05
            Sens(i,j)= 0; % found is correct
            FDR(i,j) = 0; % not find
            Acc(i,j) = 0;
        else
            ispike_detected = [];
            iinterval = [];
            add_factor = 0;
            while(length(spikes)>1)
                cur_range = spikes((spikes <= (spikes(1)+D)) & (spikes >= spikes(1)));
                [~,idx] = max(data_noised(cur_range+add_factor));
                ispike_detected = [ispike_detected, cur_range(idx)+add_factor];
                iinterval = [iinterval;[spikes(1)+add_factor-10,spikes(1)+add_factor+50]];
                if(size(ispike_detected,2) ~= size(iinterval))
                    a = 0;
                end
                sub_factor =  spikes(1) + D;
                add_factor = add_factor + spikes(1)+D;
                spikes = spikes-sub_factor;
                spikes = spikes(spikes>0);
            end
            iinterval(iinterval<1) = 1;
            iinterval(iinterval>length(data_noised)) = length(data_noised);
    %         if(i == 15)
    %             figure(10)
    %             hold on
    %             plot(data_noised);
    %             plot(thr(i,j)*ones(size(data_noised)));
    % %             scatter(spike_loc,data_noised(spike_loc),'.');
    % %             scatter(ispike_detected,data_noised(ispike_detected));hold off
    %         end
    %         hold off
            [FP,FN,TP]=locationCompare(spike_loc,iinterval,ispike_detected);
            Sens(i,j)= length(TP)/(length(TP)+length(FN)); % found is correct
            FDR(i,j) = length(FP)/(length(FP)+length(TP)); % not find
            Acc(i,j) = length(TP)/(length(TP)+length(FN)+length(FP));
        end
    end
end   
figure(4)
plot(multiplier,Acc'); hold on
for i = 1:size(Acc,1)
    [m,idx] = max(Acc(i,:));
    scatter(multiplier(idx),Acc(i,idx),'k*')
%     if(idx == 1 || idx == 2)
%         plot(multiplier(1:idx+2), Acc(i,1:idx+2),'k','LineWidth',2)
%     else
%         plot(multiplier(idx-2:idx+2), Acc(i,idx-2:idx+2),'k','LineWidth',2)
%     end
end
hold off

max_idx = -1000;
min_idx = 1000;
for i = 1:size(Acc,1)
    [m,idx] = max(Acc(i,:));
    if(m>0.8)
        if(idx > max_idx)
            max_idx = idx;
        end
        if(idx<min_idx)
            min_idx = idx;
        end
    end
end
shift = multiplier(max_idx) - multiplier(min_idx);

span = 0;
count = 0;
for i = 1:size(Acc,1)
    mul_accept = multiplier(Acc(i,:)>0.8);
    if(~isempty(mul_accept))
        count = count + 1;
        span = span + mul_accept(end) - mul_accept(1);
    end
end
span = span/count;
span7 = 0;
count = 0;
for i = 1:size(Acc,1)
    mul_accept = multiplier(Acc(i,:)>0.7);
    if(~isempty(mul_accept))
        count = count + 1;
        span7 = span7 + mul_accept(end) - mul_accept(1);
    end
end
span7 = span7/count;
s = span/span7;

factor = shift/span;

xlabel('$\alpha$')
ylabel('Accuracy')
title(['SR = ', num2str(s,2),' SSR = ',num2str(factor,2)])
% figure(2)
% plot(multiplier,FDR'); 
% figure(3)
% plot(multiplier,Sens'); 
%% Load Quiroga data
addpath(genpath('./data'))
addpath('./functions')
addpath('./utils')
clear all
number = {'05','1','15','2'};
easy_diff = {'Easy','Difficult'};
dat = [];
spike_loca = {};
N = 1e6;
for l = 1:2
    for m = 1:2
        for n = 1:4
    %     m = 1;
    %     n = z;
        dataType = 'SynData2';
        name = ['C_',easy_diff{l},num2str(m),'_noise0',number{n}];
        load(['./data/',dataType,'/',name,'.mat'])
        load('./data/LFP.mat')
%         dat=[dat;downsample(data(1:3*N)*100,3)];%+LFP(1:N);
%         spike_location = round((spike_times{1}+22)/3);
        dat=[dat;data(1:N)];%+LFP(1:N);
        spike_location = round(spike_times{1})+23;
        spike_loca = [spike_loca,spike_location(spike_location<=N-8)];
        end
    end
end
%%  detect with found parameters
D = 5;
add_factor = 0;

for i = 1:16
    data = extractMean(dat(i,:),16);
%     data =dat(i,:);
    spike_loc = spike_loca{i};
    peak_mean = mean(abs(data(spike_loc)));
    data_noise = data;
    for j = 1:length(spike_loc)
        data_noise(spike_loc(j)-3:spike_loc(j)+6) = 1000;
    end
    data_noise = data_noise(data_noise ~= 1000);
    noise_mean = mean(abs(data_noise));
    thr(i) = peak_mean * 0.5 + noise_mean * 1.5;
    spikes = find(data>thr(i));
    ispike_detected = [];
    iinterval = [];
    add_factor = 0;
    while(length(spikes)>1)
        cur_range = spikes((spikes <= (spikes(1)+D)) & (spikes >= spikes(1)));
        [~,idx] = max(data(cur_range+add_factor));
        ispike_detected = [ispike_detected, cur_range(idx)+add_factor];
        iinterval = [iinterval;[spikes(1)+add_factor-7,spikes(1)+add_factor+9]];
        if(size(ispike_detected,2) ~= size(iinterval))
            a = 0;
        end
        sub_factor =  spikes(1) + D;
        add_factor = add_factor + spikes(1)+D;
        spikes = spikes-sub_factor;
        spikes = spikes(spikes>0);
    end
    iinterval(iinterval<1) = 1;
    iinterval(iinterval>length(data)) = length(data);
%         if(j == 15)
%             figure(i)
%             plot(data);hold on
%             plot(thr(i)*ones(size(data)));
%             scatter(spike_loc,data(spike_loc),'.');
%             scatter(ispike_detected,data(ispike_detected));hold off
%         end
    [FP,FN,TP]=locationCompare(spike_loc,iinterval,ispike_detected);
    Sens(i)= length(TP)/(length(TP)+length(FN)); % found is correct
    FDR(i) = length(FP)/(length(FP)+length(TP)); % not find
    Acc(i) = length(TP)/(length(TP)+length(FN)+length(FP));
end
acc = mean(Acc)
fdr = mean(FDR)
sens = mean(Sens)

a = mean(reshape(Acc,[4,4]),2)
std(a)
mean(a)

