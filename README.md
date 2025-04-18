Time is super short. 


Plan: 

- just tokenise hacker news 
- load pre-trained Word2Vec model
- ensure we save everything on huggingface
- create a plan for specifically how we should use docker and the external server to push this code (have chatGPT do this)

- finetune model on hacker news -> save to hackernews 
- randomly sub-sample 0 score posts so that there are an equal number to posts that have 1+ posts 
- train model 
- expose as an api 


then do a bunch of visualisation 