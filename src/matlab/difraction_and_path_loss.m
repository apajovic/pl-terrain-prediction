function [J, L] = difraction_and_path_loss(cx,cy,c,m,n,x_center,y_center,s,f)
% Odredjivanje slabljenja usled difrakcije i rastojanja
% Analizira se profil terena, gde su nadmorske visine vrednosti  
% piksela u matrici za pravac odredjen centralnim elementom matrice 
% (x_center, y_center) na kom se nalazi predajnik i proizvoljnim 
% elementom matrice na kom se nalazi prijemnik. Slabljenje usled 
% difrakcije se odredjije kao slabljanje na ostrici noza na
% najistaknutijoj prepreci na zadatom pravcu.

lambda=3e8/f;
Gr=1.64; %dobitak prijemne antene
Gt=1.64; %dobitak predajne antene

% ne posmatraju se prvi i poslednji element da se visina TX i
%Rx ne bi posmatrale kao prepreka         
c=c(2:length(c)-1);
cx=cx(2:length(cx)-1);
cy=cy(2:length(cy)-1);       


%rastojanje izmedju predajnika i prijemnika,
%odnosno, izmedju tacke u centru matrice i tacke
%sa koordinatama (m,n)
%50m je rezolucija
d_tx_rx=sqrt((n-x_center)^2+(m-y_center)^2)*50;

%provera da li je c prazan niz, sto ukazuje na tacke
%susedne centralnoj tacki, odnosno, nema prepreke 
%izmedju TX i RX (ovo je nepotreban uslov kada racunam u vise tacaka) 
if isempty(c)
    J=0;    
else
%obstacle je najistaknutija prepreka, odnosno, najveca vrednost 
%piksela na trasi od centra, do (m,n) tacke
    [obstacle, index]=max(c);
%     x_obstacle=round(cx(index));
%     y_obstacle=round(cy(index));
    x_obstacle=cx(index);
    y_obstacle=cy(index);            
    %rastojanje od TX do prereke
    d_tx_obs=sqrt((x_obstacle-x_center)^2+(y_obstacle-y_center)^2)*50;
    h_tx=s(x_center,y_center)+1.5; %bilo je +30
    h_rx=s(m,n)+1.5;
    h_p=obstacle;

    if (h_tx-h_rx)>0
        x=(h_tx-h_rx)*(d_tx_rx-d_tx_obs)/d_tx_rx;
        h_ov=h_rx+x;
    else
        x=(h_rx-h_tx)*d_tx_obs/d_tx_rx;
        h_ov=h_tx+x;
    end
    %debljina Frenelove zone
    h_n=sqrt(lambda*d_tx_obs*(d_tx_rx-d_tx_obs)/d_tx_rx);

    %provera da li prepreka narusava I Frenelovu zonu i proracun
    %odgovarajuceg slabljenja
    if h_ov-h_p<=h_n
        h=h_p-h_ov;
        ni=h*sqrt(2/lambda*(1/d_tx_obs+1/(d_tx_rx-d_tx_obs)));
        if ni>=-0.78
            J=6.9+20*log10(sqrt((ni-0.1)^2+1)+ni-0.1);
        else
            J=0;
        end
    else
        J=0;
    end
end
L=20*log10(4*pi*d_tx_rx/lambda)-10*log10(Gt*Gr);
end