import lightning as L
from pathlib import Path
from model.bert import BERT_FOR_POI, BERT_Trip
from transformers import AutoConfig
from utils import MLMDataModule
from utils.util import get_data_dir, wccount

dataset_metadata = {
    'edin': {'USER_NUM':386, 'TIME_NUM': 48},
    'glas': {'USER_NUM':90, 'TIME_NUM': 48},
    'melb': {'USER_NUM':265, 'TIME_NUM': 48},
    'osaka': {'USER_NUM':40, 'TIME_NUM': 48},
    'toro': {'USER_NUM':196, 'TIME_NUM': 48},
    #new one
    'weeplaces_poi_25_length_3-15-numtraj_765': {'USER_NUM':400, 'TIME_NUM': 48},
    'weeplaces_poi_50_length_3-15-numtraj_2134': {'USER_NUM':883, 'TIME_NUM': 48},
    'weeplaces_poi_100_length_3-15-numtraj_4497': {'USER_NUM':1555, 'TIME_NUM': 48},
    'weeplaces_poi_200_length_3-15-numtraj_7790': {'USER_NUM':2537, 'TIME_NUM': 48},
    'weeplaces_poi_400_length_3-15-numtraj_12288': {'USER_NUM':3357, 'TIME_NUM': 48},
    'weeplaces_poi_801_length_3-15-numtraj_19385': {'USER_NUM':4230, 'TIME_NUM': 48},
    'weeplaces_poi_1600_length_3-15-numtraj_31545': {'USER_NUM':5276, 'TIME_NUM': 48},
    'weeplaces_poi_3200_length_3-15-numtraj_53901': {'USER_NUM':6531, 'TIME_NUM': 48},
    'weeplaces_poi_6400_length_3-15-numtraj_90248': {'USER_NUM':7819, 'TIME_NUM': 48},
    'weeplaces_poi_12800_length_3-15-numtraj_147287': {'USER_NUM':9326, 'TIME_NUM': 48},
}

def train(
    dataset_dir: Path,
    dataset_name: str,
    base_model: L.LightningModule,
):
    dataset_dir = get_data_dir() / dataset_name
    dataset_file_path = dataset_dir / 'data.csv'
    poi_vocab_file = dataset_dir / 'poi_vocab.txt'
    
    data_module = MLMDataModule(
        dataset_file_path = dataset_file_path,
        vocab_file = poi_vocab_file,
        num_user_tokens = dataset_metadata[dataset_name]['USER_NUM'],
        num_time_tokens = dataset_metadata[dataset_name]['TIME_NUM'],
    )
    #evaluator = Tester
    model_name = 'bert-base-uncased'
    config = AutoConfig.from_pretrained(model_name) 
    config.vocab_size = wccount(poi_vocab_file)
    config.hidden_size = 32
    config.num_hidden_layers = 1
    config.intermediate_size = 128
    config.num_attention_heads = 2
    
    #tester = evaluator(kw)
    #print(base_model)
    model = base_model(
        config = config,
    )
    # create pretrain_file using transition matrix by train data and then save it
    # only useful for the Flickr datasets or for the WeePlaces datasets with lower numbers of POIs.

    trainer = L.Trainer(max_epochs=1000, accelerator="auto")  # accelerator="gpu" if you want to use GPU
    trainer.fit(model, datamodule = data_module)

if __name__ == "__main__":
    
    datasets = [
    #('osaka', 47 * 5),
    #('melb', 442 * 2),
    #('edin', 634 * 2),
    #('toro', 335 * 5),
    #('glas', 112 * 10),
    #('weeplaces_poi_12800_length_3-15-numtraj_147287', 147287),
    #('weeplaces_poi_6400_length_3-15-numtraj_90248', 90248 * 3),
    #('weeplaces_poi_3200_length_3-15-numtraj_53901', 53901 * 3),
    #('weeplaces_poi_1600_length_3-15-numtraj_31545', 31545),
    #('weeplaces_poi_801_length_3-15-numtraj_19385', 19385 * 3),
    #('weeplaces_poi_400_length_3-15-numtraj_12288', 12288),
    #('weeplaces_poi_200_length_3-15-numtraj_7790', 7790),
    #('weeplaces_poi_100_length_3-15-numtraj_4497', 4497),
    #('weeplaces_poi_50_length_3-15-numtraj_2134', 2134),
    ('weeplaces_poi_25_length_3-15-numtraj_765', 765),# * 3
    ]
    for dataset_name, size in datasets:
        for model in [BERT_FOR_POI, BERT_Trip]:
            
            dataset_dir = Path().cwd() / 'data'
            train(dataset_dir, dataset_name, model)