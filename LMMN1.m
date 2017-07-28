function [ MSE ] = LMMN1( trainInput,trainTarget,testInput,testTarget,stepSizeWeightVector,delte )
[inputDimension,trainSize] = size(trainInput);
[functionError,aprioriErr,MSE,weightVector] = ...
deal(zeros(trainSize,1),zeros(trainSize,1),zeros(trainSize,1),zeros(inputDimension,1));
% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n);
    aprioriErr(n) = trainTarget(n) - networkOutput;
    functionError(n) = delte*aprioriErr(n) + ( 1 - delte )*(aprioriErr(n)^3);
    weightVector = weightVector + stepSizeWeightVector*functionError(n)*trainInput(:,n);
    %testing
    err = testTarget - (testInput'*weightVector);
    MSE(n) = mean(err.^2);
end

return


