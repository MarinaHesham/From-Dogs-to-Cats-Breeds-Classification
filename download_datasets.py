import kaggle
import kaggle.api
from zipfile import ZipFile

DATASETS = {
    'dogs':{
        'kaggle_name': 'jessicali9530/stanford-dogs-dataset',
        'dest_dir': 'dogs',
        'zip_filename': 'stanford-dogs-dataset.zip'
    },
    'cats':{
        'kaggle_name': 'ma7555/cat-breeds-dataset',
        'dest_dir': 'cats',
        'zip_filename': 'cat-breeds-dataset.zip'
    }
}

def main():
    for i in ['dogs', 'cats']:
        print(f'download {i} dataset')
        kaggle_download(DATASETS[i]['kaggle_name'], 
                        DATASETS[i]['zip_filename'],
                        DATASETS[i]['dest_dir'])


def kaggle_download(kaggle_name, zip_filename, dest_dir):
    
    kaggle.api.authenticate()
    # Checks first if the file exists, if not downloads it.
    kaggle.api.dataset_download_files(kaggle_name, path='.', unzip=False, quiet=False)

    with ZipFile(zip_filename, 'r') as zipObj:
        print(f'extracting {zip_filename} to {dest_dir}')
        zipObj.extractall(dest_dir)
    

if __name__ == "__main__":
    main()