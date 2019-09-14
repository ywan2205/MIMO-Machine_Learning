%% ====== semi-correlated train and test data set samples   

clc
clear
close all
 
Nt = 32;                            % user
N = 10;                             % length of each frame
Nr = 32;                            % BS antenna
L = 20;                          % simulation time
EbNo = 0 : 5 : 30 ;                         % SNRdB
%
M = 16;                                 % 16QAM modulation
x = randi([0, 1], Nt, N*L);            % original signal 0 or 1
s = qammod(x, M).*exp(1i*pi/4);         % modulate signal  
I = eye(Nt);                           %unit matric
for indx = 1:length(EbNo)
        indx
        x1=[];
        x2=[];
        x3=[];
        x4=[];
        x5=[];
        y6_real_and_image=[];
        
        
    for indx1 = 1 : L
            indx1
            y1=[];
            y2=[];
            y3=[];
            y4=[];
            y5=[];
            y6=[];
            
%% magnitude of the correlation 
% ====  parameter
parameter_tt(indx1) = rand(1); %random 0 to 1
% ==== random phase
%phase_tt(indx, indx1) = -pi + (2*pi)*rand(1,1);  
phase_tt(indx1) = -0.5*pi;
%%
rr = parameter_tt(indx1) * exp(1i*phase_tt(indx1));
Rr = zeros(Nr, Nr);

for ii = 1 : Nr
    for jj = 1 : Nr
        if (ii == jj)
        	Rr(ii, jj) = 1;
        elseif (ii < jj)
            Rr(ii, jj) = power(rr, jj - ii);
        elseif (ii > jj) 
            Rr(ii, jj) = conj(Rr(jj, ii));  %gong e 
        end
    end
end
%%     
        G = ( randn(Nr, Nt) + 1i*randn(Nr, Nt) ) ./ sqrt(2);      % Rayleigh channel     
        sigmal = Nt * 1 / (10.^(EbNo(indx)/10)) ;                 % Gaussian white noise standard deviation       
        sig = 1 / (10.^(EbNo(indx)/10));         
        n = sigmal * (randn(Nr, N)+1i*randn(Nr, N));            % noise
        y = G * s(:, N*(indx1-1)+1:(N*indx1)) + n;              % receive signal
            
%% consider semi kroneckor correlation channel model       
        H2 = sqrtm(Rr) * G ; 
        y6 = H2 * s(:, N*(indx1-1)+1: (N*indx1)) + n;   
        Data{indx,indx1} = y6;

%% separate real and image part     
        y6_save_real(:, :) = real(y6);
        y6_save_Imag(:, :) = imag(y6);
        for i = 1 : N
            y6_save_real_image(indx1, :, 2*i-1) = y6_save_real(:, i);
            y6_save_real_image(indx1, :, 2*i) =  y6_save_Imag(:, i);  
        end 

    end
    % save data to files
    test_filename = ['train_and_test_data',num2str(indx),'.mat'];
    save(test_filename,'y6_save_real_image')
    test_filename = ['label',num2str(indx),'.mat'];
    save(test_filename,'parameter_tt')
end




        
        
        
        
        
