# NetMedGPT

## Prepare environment
First clone the respositiy and use install the envinomtm
```
git clone https://github.com/faren-f/NetMedGPT.git
cd NetMedGPT
conda env craete -f chatnetmedgpt.yml
```

## Run ChatNetMedGPT
To run the model and get the result use the command below
```
python ChatNetMedGPT/netmedgpt_llm.py --user_text "user_text"
```
```"user_text"``` is the user query. For example
```
python ChatNetMedGPT/netmedgpt_llm.py --user_text "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions"
```
The output 
