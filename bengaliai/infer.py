import numpy as np
import pandas as pd
import torch


def predict(model, data_loader, input_key: str, names: list, device=None) -> dict:
    model.eval()
    
    batch_y_predictions = []
    
    with torch.no_grad():
        for data in data_loader:
            inputs = data[input_key].to(device)
                      
            outputs = model(inputs)
            
            batch_y_predictions.append(outputs)
            
    y_pred_prob = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*batch_y_predictions)]
    y_pred = [np.argmax(a, axis=-1) for a in y_pred_prob]
    
    return {n:v for n, v in zip(names, y_pred)}


def get_submission(model, data_loader, input_key: str, names: list, device=None) -> pd.DataFrame:
    y_pred = predict(model, data_loader, input_key, names, device)
    arrays = [y_pred[k] for k in names]
    flatten_pred = sum(zip(*arrays), ())

    row_id = []
    for i in range(pc[0].shape[0]):
        row_id.extend(['Test_%s_%s' %(i, n) for n in names])

    return pd.DataFrame({'row_id': row_id, 'target': flatten_pred})

