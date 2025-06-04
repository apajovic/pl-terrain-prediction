function [wrap_img_tensor] = wrap_img(dir_wrap, tensor_unwrap, img_size, base, indikator)
% Funkcija koja unwrap slike vraca u originalni radijalni oblik
%
% Ulazi:
% - dir_wrap - direktorijum u kojem cemo cuvati kreirane slike
% - PL_pred_unwrap - tensor sa slikama za obradu
% - base - osnova za string u nazivu slike, da li je PL ili ter

% provera da li folder u kojem cemo cuvati kreirane slike postoji i 
% njegovo kreiranje ukoliko ne postoji
if ~exist(dir_wrap, 'dir')
    mkdir(dir_wrap);  
end

% obrada svake slike iz tensora - vracanje u radijalni oblik
for i = 1: size(tensor_unwrap)
    unwrap_img = squeeze(tensor_unwrap(i,:,:));
    center = [size(unwrap_img,1)/2, size(unwrap_img,2)/2];
    wrp_img = radial_wrap(unwrap_img, img_size, center);
    wrap_img_tensor(:,:,1,i) = wrp_img;
end

% Cuvanje unwrap slika 
for i=1:size(wrap_img_tensor,4)
    img_name = sprintf('%s%s%03d.png', base, '_wrap', i);
    fileName = fullfile(dir_wrap, img_name);

    % zaokruzujemo vrednosti matrice na cele brojeve jer su to pikseli
    if indikator
        img_round = uint8(wrap_img_tensor(:,:,1,i)*255);
    else
        img_round = uint8(wrap_img_tensor(:,:,1,i));
    end
    
    img_res = imresize(img_round, [256 256]);
    imwrite(img_res, fileName);
end

end