A=rand(10,10);
x=[1,2,3,4,5,6,7,8,9,10];
v = zeros(10,1);
for i = 1:10
  for j = 1:10
    v(i) = v(i) +A(i,j)*x(j);
  end
end 
v
%fprintf(" a ");
%a=A*x;
%a
%fprintf(' b ');
%b=Ax;
%b
%fprintf(' c ');
%c=x'*A;
%c
%fprintf(' d ');
%d=sum(A*x);
%d
X=rand(7,7);
X
B= X.^2;
B
B1=X^2;
B1
C=X+1;
C
D=X/4;
D
