function [U,D,V]=SVD(A)
 
[m,n] = size(A);
B=A;
if m<n %仅适用于m>n的情况 如果m<n 现将矩阵转置再分解
    A = A'; 
end
[v,D] = eig(A'*A);%D特征值
 V = flip(v,2);
 si = size(D);
 n=0;
  D(D<1e-10)=0;%很关键的一个限制，将小于某一个数的特征值设为0 否则无法判断非零特征值的数量
 for i=1:si(1)
     if  D(i,i)>0
        n=n+1; 
     end
 end
  si = size(V);
  V1 = zeros(si(1),n);
 
%  for i=1:n
%       V1(:,i) = V(:,i); %非零特征值向量
%  end
  V1(:,1:n) = V(:,1:n); %非零特征值向量
  si = size(D); %分离零和非零特征向量
  if n==si(1)%无零特征值
       V = V1;
  else 
       V2 = V(:,n+1:si(1));
       V = [V1,V2];
  end
 h = zeros(n,n);%特征值对角矩阵
 for i=1:n
     h(i,i) = D(si(1)-i+1,si(1)-i+1)^0.5; %奇异值矩阵
 end
 h = inv(h);
 U1 = A*V1*h;
 
 
[v1,D1] = eig(A*A');%D特征值 V特征向量
si = size(D1);
m = zeros(si(1),1);
for i=1:si(1)
    m(i) = D1(i,i);
end
 m(m<1e-10)=0; %很关键的一个限制，将小于某一个数的特征值设为0，否则无法判断非零特征值的数量
%求U2
% m = eig(A*A'); %求A*A'特征值个数
si = size(m);
m = flip(m); %从大到小排列
n=0;%奇异值个数
 for i=1:si(1)%计算零特征向量
     if m(i)==0
         n = n+1;
     end
 end
 si = size(v1);
 U2 = zeros(si(1),n);
 U2 = v1(:,1:n);
 U = [U1,U2];
 D = zeros(size(A));
 si = size(m);
 for i=1:si(1)-n
     D(i,i) = m(i)^0.5;
 end
[m,n] = size(B);
if m<n %A = UDV'  A' = VD'U'
   u = zeros(size(U));
   u = U;
   U = V;
   V = u;
   D = D';
end
