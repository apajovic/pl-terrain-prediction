% Kod koji na osnovu slika terena i predikcije treba da napravi njihovu
% unwrap verziju i sacuva je kao slike koje koristi za testiranje u Python-u. 
% Nakon toga, preuzete predikcije u formi tensora iz Python-a se ovde 
% obradjuju za dalju statistiku.
% Koriscenje funkcije
% - unwrap_img / radial_unwrap
% - wrap_img / radial_wrap
% - applyCircularMask
%% kreiranje unwrap slika
% -----------------------------NAPOMENA------------------------------------
% U ovom delu koda se radi priprema podataka koji ce se koristiti u
% Pajtonu za testiranje modela - a to su unwrap slike terena i PL-a koje su
% sacuvane u odgovarajucim folderima kao .png slike

% direktorijum u kom se se cuvati teren i PL unwrap slike
dir_ter_unwrap = fullfile('C:\','Users','telekom','Desktop','doktorat','unwrap_wrap_model','ter_unwrap');
dir_PL_unwrap_orig = fullfile('C:\','Users','telekom','Desktop','doktorat','unwrap_wrap_model','PL_unwrap_orig');

% putanje na kojima se nalaze originalne slike terena i PL-a
folder_path_ter = 'C:\Users\telekom\Desktop\doktorat\unwrap_wrap_model\ter_wrap';
folder_path_PL = 'C:\Users\telekom\Desktop\doktorat\unwrap_wrap_model\PL_wrap_orig'; 

% koliko koraka uzimamo po krugu i po radijusu 
num_angles = 500;
num_radii = 300;
img_size = [256 256];

% poziv funkcije koja 
% - na osnovu originalnih slika kreira unwrap slike sa zadatim parametrima, 
% - radi resize na 256x256,
% - cuva ih kao png sa pikselima u opsegu 0-255,
% - kao rezultat vraca tensore kod kojih su elementi realni brojevi u opsegu 0-255
ter_tensor_unwrap = unwrap_img(dir_ter_unwrap, folder_path_ter, num_angles, num_radii, img_size,'ter');
PL_tensor_unwrap_orig = unwrap_img(dir_PL_unwrap_orig, folder_path_PL, num_angles, num_radii, img_size,'PL');

%% statistika i vizuelizcija predikcije
% -----------------------------NAPOMENA------------------------------------
% - U ovom delu koda se radi preuzimanje rezultata predikcije i targeta iz
% Pajtona nad kojima je model testiran (i koji su vraceni u originalni
% opseg vrednosti za PL). 
% Rezultati predikcije se porede sa targetima iz Matlaba (inicijalni 
% proracuni u radijalnom formatu) i radi se njihova vizuelizacija. 
% Dodatno, radi se statistika i za unwap format radi poredjenja sa
% rezultatima iz Python-a

% Direktorijum u kom ce se cuvati PL predikcije u wrap formi
dir_PL_wrap_pred = fullfile('C:\','Users','telekom','Desktop','doktorat','unwrap_wrap_model','PL_wrap_pred');

% ucitavanje rezultata predikcije iz Pajtona - rezultati predikcije kao i
% targeti su vec u Python-u vraceni u originalni opseg PL vrednosti
data = load('predictions_and_targets.mat');
predictions = data.predictions;  
targets = data.targets;

% Python predikcija u wrap formi i kreiranje tensora sa wrap pred slikama
% Primenjujem i kruznu masku da bih van kruga stavila NaN vrednosti sto ce
% mi biti znacajno kasnije kod proracuna za evaluaciju modela
indikator = 0; % da li je slika u opsegu od 0 - 1 (1) ili 0 - 255 (0)
PL_wrap_pred_py = wrap_img(dir_PL_wrap_pred, predictions, img_size, 'PL', indikator);
for i = 1: size(PL_wrap_pred_py, 4)
    img = applyCircularMask(PL_wrap_pred_py(:,:,1,i));
    PL_wrap_pred(:,:,1,i) = img;
end

% - ucitavanje tensora sa originalnim PL vrednostima
% - primenjivanje kruzne maske
load_tensor = load('PL_tensor.mat');
PL_wrap_orig = load_tensor.PL_tensor;
for i = 1:length(PL_wrap_orig)
    PL_wrap_orig_masked(:,:,1,i) = applyCircularMask(PL_wrap_orig(:,:,1,i));
end
%%
% Odredjujemo vrednosti statistickih parametara na osnovu:
% - PL_wrap_pred - tensor sa rezultatima predikcije PL-a iz Pajtona (wrap vrednosti)
% - PL_wrap_orig_masked - tensor sa proracunima PL-a
% Prilikom izracunavanja, potrebno je uzeti u obzir samo one piksele koji
% su unutar kruga u kom su PL vrednosti

error_tensor =  PL_wrap_pred -  PL_wrap_orig_masked;
fprintf('MEAN = %.2f\n', mean(error_tensor(:),"omitmissing"));
fprintf('MSE = %.2f\n', mean(error_tensor(:).^2, "omitmissing"))
fprintf('RMSE = %.2f\n', sqrt(mean(error_tensor(:).^2, "omitmissing")));
fprintf('STD =  %.2f\n', std(error_tensor(:),"omitmissing"));

% vizualizacija
sample_indxs = randsample(size(PL_wrap_orig_masked, 4), 3);  % uzmi 3 nasumična indeksa
figure;

for i = 1 : 3
    idx = sample_indxs(i);
    
    % pronalazanje min i max vrednosti unutar kruga sa PL vrednostima
    img_pred = PL_wrap_pred(:,:,1,idx);
    img_trgt = PL_wrap_orig_masked(:,:,1,idx);
    pixels = [img_pred(:); img_trgt(:)];
    pixels = pixels(~isnan(pixels));
    max_pix_value = max(pixels);
    min_pixel_value = min(pixels);

    % Target (original) - leva kolona
    subplot(3, 2, 2*i - 1);
    imagesc(PL_wrap_orig_masked(:,:,1,idx));
    title(sprintf('Target - num %d', idx));
    clim([min_pixel_value, max_pix_value]); c =colorbar;
    c.Label.String = 'Path Loss [dB]'; 

    % Predikcija - desna kolona
    subplot(3, 2, 2*i);
    imagesc(PL_wrap_pred(:,:,1,idx));
    title(sprintf('Predikcija - num%d', idx));
    clim([min_pixel_value, max_pix_value]); c =colorbar;
    c.Label.String = 'Path Loss [dB]'; 
end

%% 
% rmse_ = zeros(size(predictions,1),1);
% std_ = zeros(size(predictions,1),1);
% mean_ = zeros(size(predictions,1),1);
% for i = 1:size(predictions,1)
%     img_pred = squeeze(predictions(i,:,:));
%     img_trgt = squeeze(targets(i,:,:));
%     err_d = img_pred-img_trgt;
%     std_(i) = std(err_d(:));
%     mean_(i) = mean(err_d(:));
%     rmse_(i) = sqrt(mse(err_d(:)));
% end
% 
% r = mean(rmse_)
% s= mean(std_)
% m = mean(mean_)
% 
% MSEs = zeros(length(PL_wrap_orig_masked),1);
% RMSEs = zeros(length(PL_wrap_orig_masked),1);
% STDs = zeros(length(PL_wrap_orig_masked),1);
% MEANs = zeros(length(PL_wrap_orig_masked),1);
% 
% for i = 1 : size(targets,1)
%     err = PL_wrap_orig_masked(:,:,1,i) - PL_wrap_pred(:,:,1,i);
%     %figure, imagesc(err), colorbar;
%     MSEs(i) = mse(err(:));
%     RMSEs(i) = sqrt(mse(err(:)));
%     STDs(i) = std(err(:));
%     MEANs(i) = mean(err(:));
% end

