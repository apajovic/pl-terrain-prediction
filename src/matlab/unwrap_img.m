function [unwrap_tensor] = unwrap_img(dir_unwrap, folder_path, num_angles, num_radii, img_size, base)
% Funkcija koja na osnovu originalnih kreira unwrap slike sa zadatim
% parametrima, radi resize na 256x256 i cuva ih kao .png sa pikselima u
% opsegu 0-255, takodje cuva i tensore (jer proracun dugo traje, pa da mi
% ostanu za svaki slucaj sacuvani). Tensor sadrzi podatke dobijene
% interpolacijom, znaci, oko vrednosti u opsegu 0-255, ali realni brojevi,
% nisu int jer se radi interpolacija
% 
% Ulazi:
% - dir_unwrap - direktorijum u kojem cemo cuvati kreirane slike
% - folder_path - putanja do foldera u kom se nalaze originalne slike
% - num_angles - broj koraka po uglu
% - num_radii- broj koraka po radijusu
% - base - osnova za string u nazivu slike, da li je PL ili ter

% provera da li taj folder u kojem cemo cuvati kreirane slike postoji i 
% njegovo kreiranje ukoliko ne postoji
if ~exist(dir_unwrap, 'dir')
    mkdir(dir_unwrap);  
end


%lista sa slikama iz foldera sa originalnim slikama koje cemo da
%unwrap-ujemo
file_list = dir(fullfile(folder_path, '*.png')); 

for k = 1:length(file_list) % (ovde menjam kad hocu da kontrolisem br slika za test)
    file_name = file_list(k).name;
    full_path = fullfile(folder_path, file_name);
    
    img = imread(full_path);       
    center = [size(img,1)/2, size(img,2)/2];

    % Unwrap
    unwrap_img = radial_unwrap(img, num_angles, num_radii, center);   
    unwrap_tensor(:,:,1,k)=unwrap_img;
end

% Cuvanje unwrap slika 
for i=1:size(unwrap_tensor,4)
    img_name = sprintf('%s%s%03d.png', base, '_unwrap', i);
    fileName = fullfile(dir_unwrap, img_name);

    % zaokruzujemo vrednosti matrice na cele brojeve jer su to pikseli
    img_round = uint8(unwrap_tensor(:,:,1,i));

    if size(img_round,1)~=256 && size(img_round,2)~=256
        img_res = imresize(img_round, img_size);
        imwrite(img_res, fileName);
    else
        imwrite(img_round, fileName);
    end  
end

%save('unwrap_tensor', 'unwrap_tensor')
end