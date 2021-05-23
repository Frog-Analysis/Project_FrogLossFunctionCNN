function y=renyi_entro(x,q)

% As q approaches zero, the R¨¦nyi entropy increasingly weighs all possible events
% more equally, regardless of their probabilities. In the limit for ¦Á ¡ú 0, the R¨¦nyi entropy 
% is just the logarithm of the size of the support of X. The limit for ¦Á ¡ú 1 is the Shannon entropy. 
% As ¦Á approaches infinity, the R¨¦nyi entropy is increasingly determined by the events of highest probability.

[M,N]=size(x);
if M < N
    x = x';
    y=zeros(1,M);
else
    y=zeros(1,N);
end
for n=1:min(M,N)
    % for n=1:N
    y(1,n)=log(sum(x(:,n).^q))/(1-q);
end
%[EOF]