import yaml
import argparse
import numpy as np
from pathlib import Path
from scripts.utils import DataLoader, WORK_PATH
from models.transformer import Transformer
from models.genfactor import GenFactor
from models.mlp import MLP
import pandas as pd
import torch
import gc
import time
import logging
import json
from tqdm import tqdm

logging.basicConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_and_predict(model):
    logging.info(f"Start inferencing and predicting for model <{model.name}>")
    gBar = tqdm(
            range((len(model.dl.date) - 1260)//252),
            colour="red",
            desc=f"{model.name} Inferencing & Predicting",
        )
    pred_result = []
    label_result = []
    date_result = []
    for g in gBar:
        try: # ugly but works
            model.reset_weight()
            model.refit()
            gBar.set_postfix({"Year": int(model.valid_period[0][:4])})
            model.train_model()
            pred, label, date = model.predict()
            pred_result+=pred
            label_result+=label
            date_result+=date
        except:
            pass
    pred_result = pd.concat(pred_result,axis=1)
    label_result = pd.concat(label_result, axis=1)
    pred_result.columns, label_result.columns = date_result, date_result
    pred_result.to_csv(model.save_path+'prediction'+f'/{model.name}.csv')
    label_result.to_csv(model.save_path+'label'+f'/{model.name}.csv')

    # cal metrics
    oos_prediction_R2 = 1 - (np.nansum((pred_result - label_result)**2)/ np.nansum(label_result**2))
    IC = []
    for col in pred_result.columns:
        IC.append(np.corrcoef(pred_result[col].dropna(), label_result[col].dropna())[1,0])
    
    # save IC & R_square to json
    logging.info(f"Saving IC & R_square to json")
    p = time.localtime()
    time_str = "{:0>4d}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}".format(
        p.tm_year, p.tm_mon, p.tm_mday, p.tm_hour, p.tm_min, p.tm_sec
    )
    filename = f"IC_R_squares/{time_str}.json"
    obj = {
        "models": model.name,
        "R2_pred": oos_prediction_R2,
        "IC_pred": np.nanmean(IC),
    }

    with open(filename, "w") as out_file:
        json.dump(obj, out_file)
    logging.info(f"Model name <{model_name}> finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Model",
        type=str,
        default="mlp transformer"
    )
    args_list = parser.parse_args()
    args = {}
    
    for args_name in args_list.Model.split(' '):
        file_path = WORK_PATH / Path('configs') / Path(args_name+'.yml')
        with open(file_path, 'r') as file:
            args[args_name] = yaml.safe_load(file)
            
    logging.info("Start loading data and model...")
    for model_name in args:
        logging.info(f"Model name <{model_name}>")
        logging.info(f"parameters: {args[model_name]}")
        if model_name == 'transformer':
            dl = DataLoader(args[model_name], device)
            model = Transformer(model_name, dl, args[model_name])
            
        elif model_name == 'mlp': 
            dl = DataLoader(args[model_name], device)
            model = MLP(model_name, dl, args[model_name])
        
        elif model_name == 'genfactor':
            dl = DataLoader(args[model_name], device)
            model = GenFactor(model_name, dl, args[model_name])
        
        inference_and_predict(model)