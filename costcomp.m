function [J,grad]=costcomp(X,y,thetaunrolled,lambda,a,b,c)  %4,10,3  without bias
  theta1=reshape(thetaunrolled(1:(b*(a+1))),b,a+1);  %10x5
  theta2=reshape(thetaunrolled(((b*(a+1))+1):end),c,b+1);  %3x11
  a1=X';   %5x150
  z2=theta1*a1;  %10*150
  a2=[ones(1,length(X));sigmoid(z2)];  %11x150
  z3=theta2*a2;  %3x150
  a3=sigmoid(z3);   %3x150
  J=0;
  m=length(X);
  for i=1:m
    yeff=zeros(c,1);
    yeff(y(i))=1;
    J=J+sum(((-yeff).*log(a3(:,i)))-((1-yeff).*log(1-a3(:,i))));
  end
    
    J=J/m;
    tt1=theta1(:,2:end);
    tt2=theta2(:,2:end);
    ttt=[tt1(:);tt2(:)];
    
    J=J+(lambda/m)*sum(ttt.^2);
    
    delta1=zeros(size(theta1));
    delta2=zeros(size(theta2));
    
    
    for j=1:m
      
       yeff=zeros(c,1);  %3x1
       yeff(y(j))=1;
      a1=X(j,:)';  %5x1
      z2=theta1*a1;  %10x1
      a2=[1;sigmoid(z2)];   %11x1
      z3=theta2*a2;  %3x1
      a3=sigmoid(z3);  %3x1
      
      
      sigma3=a3-yeff; %3x1
      sigma2=(theta2(:,2:end)'*sigma3).*a2(2:end).*(1-a2(2:end));  %10x1
      
      
      delta1=delta1+sigma2*a1';  %10x5
      delta2=delta2+sigma3*a2';  %3x11
      end
      
      
      vert1=theta1;
      vert1(:,1)=0;
      vert2=theta2;
      vert2(:,1)=0;
      D1=(1/m)*delta1+(lambda/m)*vert1;
      D2=(1/m)*delta2+(lambda/m)*vert2;
    grad=[D1(:);D2(:)];
    
    end
    
    
      
      
      
      
      
      
      
    
    
    
    
  
  