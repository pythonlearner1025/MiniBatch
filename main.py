from pipeline.pipeline import Pipe
from sklearn.preprocessing import StandardScaler
from pipeline.minidescent.fast import MiniBatch
#from pipeline.minidescent.fast_clone import MiniBatch
model = Pipe([StandardScaler(), MiniBatch(batch=128, epoch=500, hidden_layer_num=1, hidden_layer_size=40, output_layer_size=10,
                                           alpha=0.6, Lambda=0.3)])
# add LAMBDA function to optimize lambda.
# what did we learn today:
# the deeper the layer, the greater the step must be... set batch at 500
# at hidden layer num = 4, epoch=100,  alpha = 5.5 is the max before divergence
# at hl = 4, epoch=300, alpha = 3 seems to work best
# at hl = 5, epoch=500, alpha = 3 works
# at hl = 6, epoch=500, alpha= 4 worksoo
X,y = model.load("ex3data1.mat")
model.fit(cv=True)
model.score()
ans = model.predict(X[0])
print(ans)






# gpu checker:C:\Windows\System32\DriverStore\FileRepository\nvmiui.inf_amd64_ab6fcb98a4599930\nvidia-smi





