function [theta1,theta2]=train(X,y,noclass)   %X with ones
  a=(size(X,2))-1;
  b=400;  %HIDDEN LAYER SIZE
  c=noclass;
  lambda=0.0; %LAMBDA VALUE
  epslon=0.001;  %EPSLON VALUE FOR INITIALIZATION
 

  initheta1=rand(b,a+1)*(2*epslon)-epslon;
  initheta2=rand(c,b+1)*(2*epslon)-epslon;
  
  itu=[initheta1(:);initheta2(:)];
  
  options=optimset('gradobj','on','maxiter',100);
  ftu=fmincg(@(t)(costcomp(X,y,t,lambda,a,b,c)),itu,options);
  
  finaltheta1=reshape(ftu(1:(b*(a+1))),b,a+1);
  finaltheta2=reshape(ftu((b*(a+1))+1:end),c,b+1);
  
  theta1=finaltheta1;
  theta2=finaltheta2;
  end
  
  
  