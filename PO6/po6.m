I = rgb2gray(imread('peppers.png')) ;
[ny,nx] = size(I) ;
[X,Y] = meshgrid(1:nx,1:ny) ;
figure
ax = gca;
surf(X,Y,I)
shading interp 
colormap(gray)
set(gca,'Ydir','reverse')