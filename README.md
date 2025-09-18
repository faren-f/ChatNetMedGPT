# NetMedGPT

## Prepare environment
1. First clone the respositiy and use install the envinomtm
```
git clone https://github.com/faren-f/NetMedGPT.git
cd NetMedGPT
conda env craete -f chatnetmedgpt.yml
```

2. Download the data and model parameters
```
wget https://cloud.uni-hamburg.de/public.php/dav/files/AoHLe9ARTC6Z3tL/?accept=zip
unzip index.html?accept=zip
mv data/NetMedGPT.pt model
rm index.html?accept=zip
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
The output is saved as a ```.csv``` file at ```data/user_response```.
