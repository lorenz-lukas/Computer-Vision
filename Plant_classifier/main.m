
%%%% Decision Tree
t = fitctree(meas(:,1:2), species,'PredictorNames',{'SL' 'SW' });
[grpname,node] = predict(t,[x y]);
gscatter(x,y,grpname,'grb','sod')
view(t,'Mode','graph');
dtResubErr = resubLoss(t)
cvt = crossval(t,'CVPartition',cp);
dtCVErr = kfoldLoss(cvt)


resubcost = resubLoss(t,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all');
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')
pt = prune(t,'Level',bestlevel);
view(pt,'Mode','graph')

[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off

cost(bestlevel+1)