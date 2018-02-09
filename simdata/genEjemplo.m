function [ y ] = genEjemplo(picos,limite, nfile) 
m = (limite/(picos+1));
u = m + rand();
y=0;
s = (picos/1.1)*(rand()+0.08);
x = (0:1:limite);
while(u<limite)
    y = y + normpdf(x,u,s);
    u = u + m + 2*rand(); 
    s = (picos/1.1)*(rand()+0.08);
end;
yx=10*y;
m=max(yx);
Yx=yx/m;
%figure
plot(x,Yx);xlabel('x1'); ylabel('x2');
%Salida = (picos);
Yx = horzcat(Yx,picos);
dlmwrite(nfile, Yx, '-append', 'delimiter', ',', 'precision','%0.6f');
%dlmwrite('Test.csv', Yx, '-append', 'delimiter', ',', 'precision','%0.6f');
end
