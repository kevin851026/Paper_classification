-------------------------------------------------------
File description
lib.txt: 	requirement 
category.json: 	preindex of feature Categories
rnn.py: 	model sturcture of Roberta + RNN + Linear Classifier
rnn_attn.py: 	model sturcture of Roberta + RNN + Attention + Linear Classifier
rnn_cg.py: 	model sturcture of Roberta + RNN with feature:Category + Linear Classifier
model_ensemble.py  ensemble result from pretrained model
train.csv:	training data
test.csv:	testing data
task2_sample_submission: sample file to creat result file
out.csv:	result file which can upload to website directly
-------------------------------------------------------
Usage
1.install all modules in lib.txt by pip instruction
2.Train the model with
    python rnn.py
    python rnn_attn.py
    python rnn_cg.py

3.New the trained model into model_ensemble.py on this form
    model_name= ['rnn.pkl','rnn_attn.pkl','rnn_cg.pkl', ...]
    model_structure = ['rnn','rnn_attn','rnn_cg', ...]
    use_categorys = [False,False,True, ...]
4.Run model_ensemble.py to get final answer: out.csv