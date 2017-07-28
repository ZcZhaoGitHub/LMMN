function [ trainInput ,trainTarget, testInput,testTarget,Xnp ] = ...
         systemIdentify( inputDimension,np,trainSize,testSize )
% system identify
[ a,b,X0 ] = deal([ 1 0 0 0 0 ],...
             [ 0.227 0.460 0.668 0.460 0.227 ],[ 0 0 0 0]);
% noise        
Xnp = sqrt(np)*randn( 1, 2*trainSize ) ;
Xn = 1*randn( 1 , 2*trainSize ) + Xnp;
Yn = filter( b , a , Xn );
noise = sqrt(np)*randn( 1 , length(Yn));
[ diff_init , diff_desire ] = deal( [ X0 Xn ] , ( Yn + noise ) );

[ trainInput , testInput ] = ...
              deal( zeros(inputDimension,trainSize),zeros(inputDimension,testSize));
          
for k = 1:trainSize
    trainInput(:,k) = diff_init( k : k+inputDimension-1 )';
end

for k = 1:testSize
    testInput(:,k) = diff_init( trainSize+k : trainSize+k+inputDimension-1 )';
end

trainTarget = diff_desire( 1 :trainSize )';
testTarget = diff_desire( trainSize + 1 : trainSize + testSize )';

end

