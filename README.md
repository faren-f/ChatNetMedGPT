# NetMedGPT

## Prepare environment
1. First clone the repository and install the environment
```
git clone https://github.com/faren-f/ChatNetMedGPT.git
cd ChatNetMedGPT
conda env create -f chatnetmedgpt.yml
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
python netmedgpt_llm.py --user_text "user_text"
```
```"user_text"``` is the user query. For example
```
python netmedgpt_llm.py --user_text "for diabetes with egfr mutation what is the best treatment and what are the adverse drug reactions"
```
The output is saved as a ```.csv``` file at ```data/user_response```.
