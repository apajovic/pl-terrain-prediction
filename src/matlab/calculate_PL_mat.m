function [PL_matrix] = calculate_PL_mat(s)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

PL_matrix=zeros(size(s));
[num_rows, num_cols]=size(s);

f=700e6; %Hz

%odredjivanje osetljivosti prijemnika
% c_i=20; %odnos signal/interferencija [dB]
% k=1.38*10^(-23); % Bolcmanova konstanta
% f_s=9; %faktor suma za prijemnik [dB]
% T=290; % temperatura [K]
% B=15000; % razmak podnosioca u LTE [Hz]
% n_n=10*log10((10^(f_s/10)*k*T*B)/(10^(-3)));
% rx_min=n_n+c_i; % osetljivost prijemnika
% rx_min=-100; %dBm

%koordinate centralnog elementa u matrice na kom se nalazi Tx
x_center=128;%100; 300
y_center=128;%100; 300

for m = 1:num_rows
    for n= 1:num_cols
        if (n==x_center) && (m==y_center)
            % za maksimalnu vrednost nivoa snage signala se ne uzima
            % predajna snaga, nego snaga na ivici rezolucijskog elementa,
            % odnosno, na rastojanju od 25 m; racunato je samo propagaciono
            % slabljenje
            PL_rx=-(10*log10(1.64)+10*log10(1.64)+20*log10((3e8/f)/(4*pi*25)));
            PL_matrix(m,n)=PL_rx;
        else
        %koordinate piksela za koji se odredjuje profil
        %kako je koordinatni sistem slike takav da je m=y i 
        %n=x, tada se u fji improfile element matrice (m,n) posmatra
        %kao (n,m), tj. xi=n, a yi=m
        [cx,cy,c]=improfile(s,[x_center n],[y_center m]);
        
        %slabljenje usled difrakcije
        [J, L]=difraction_and_path_loss(cx,cy,c,m,n,x_center,y_center,s,f);  
       
        %snaga na prijemu
        PL_rx=L+J;
        
        PL_matrix(m,n)=PL_rx;
        end
    end
end 

end

