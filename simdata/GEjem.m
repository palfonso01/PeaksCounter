% Max number of peaks
max=10;
% Number of examples
Nex = %max*400; %100;
% random indices between 1-10
x = int32((max-1).*rand(Nex,1)+1);
for i=1:Nex
    genEjemplo(double(x(i)),255, 'Training.csv');
    %genEjemplo(double(x(i)),255, 'Test80.csv');
end