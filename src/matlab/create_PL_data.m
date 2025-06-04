% U ovom kodu na osnovu generisanih slika sa preprekama (0 osnova i linije 
% proizvoljnih visina) generisu matrice sa odgovarajucim PL proracunima i
% cuvaju u folderu
% Cuva se PL tensor sa realnim (nenormalizovanim) vrednostima propagacionog
% slabljenja

% direktorijum u kom se se cuvati PL slike
dir_PL = fullfile('C:\','Users','telekom','Desktop','doktorat','unwrap_wrap_model','PL_wrap_orig');

% provera da li taj folder postoji i njegovo kreiranje ukoliko ne postoji
if ~exist(dir_PL, 'dir')
    mkdir(dir_PL);  
end

% Preuzimanje već generisanih slika terena, njihovo pretvaranje u matricu i
% prosledjivanje funkcijama za PL proračune 
% NAPOMENA - ovo vazi samo za ove vestacki generisane terene koji svakako
% imaju neki int vrednost koja odgovara vrednosti piksela, sa realnim
% terenom bismo ove proracune radili na osnovu matrica sa realnim
% vrednostima
folder_path = 'C:\Users\telekom\Desktop\doktorat\pp_model_v3\test\generisane_slike';  
file_list = dir(fullfile(folder_path, '*.png'));  % može i *.jpg, *.tif itd.

for k = 1:length(file_list)
    file_name = file_list(k).name;
    full_path = fullfile(folder_path, file_name);
    
    img = imread(full_path);        % učitavanje slike
    img = double(img);              % konverzija u realne brojeve
    
    PL_matrix = calculate_PL_mat(img);    
    PL_tensor(:,:,1,k)=PL_matrix;
end

min_ = min(PL_tensor(:));
max_ = max(PL_tensor(:));
for i=1:size(PL_tensor,4)
    fileName = fullfile(dir_PL, sprintf('PL_%3d.png', i));
    PL_mat_scaled = uint8((PL_tensor(:,:,1,i)-min_)/(max_-min_)*255);
    imwrite(PL_mat_scaled, fileName);
end