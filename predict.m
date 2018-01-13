function ypred=predict(X,theta1,theta2)  %X with ones
  a1=X';   %5x150
  z2=theta1*a1;  %10*150
  a2=[ones(1,length(X));sigmoid(z2)];  %11x150
  z3=theta2*a2;  %3x150
  a3=sigmoid(z3);   %3x150
  yy=zeros(size(X,1),1);
  for i=1:size(X,1)
    [m,yy(i)]=max([a3(:,i)]);
    end
    ypred=yy;
    end
    