import os
import json
import shutil
import kaggle


def kaggle_upload_pretrained_model(user: str, filepath: str, dataset_name: str, version_notes: str = None):
    defalut_version_notes = 'New version of model params'
    version_notes = version_notes or defalut_version_notes
    
    os.makedirs(dataset_name, exist_ok=True)
    shutil.copyfile(filepath, dataset_name + '/' + 'model.pth')
    
    dataset_kaggle_id = f"{user}/{dataset_name}"
    metadata = {
        "title": f"{dataset_name}", 
        "id": dataset_kaggle_id,
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(f'./{dataset_name}/dataset-metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    if kaggle.api.dataset_status(dataset_kaggle_id):
        print(f'Dataset {dataset_name} exists. Create new version...')
        result = kaggle.api.dataset_create_version(dataset_name, version_notes, convert_to_csv=False, delete_old_versions=False)
    else:
        print(f'Create dataset {dataset_name}...')
        result = kaggle.api.dataset_create_new(dataset_name)
        
    print('Status:', result.status)
    
    if result.status == 'error':
        print('Error:', result.error)
    
    return result

