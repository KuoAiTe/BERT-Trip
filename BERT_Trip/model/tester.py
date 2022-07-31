import torch
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from transformers import BertTokenizer
from util import calc_F1, calc_pairsF1, true_f1, true_pairs_f1, bleu, datetime_to_interval
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from typing import Optional, Tuple, Any

class ResultManager:
    def __init__(self, verbose = True):
        self.result = {}
        self.test_types = ['one_by_one', 'all_in_once']
        self.result_dir = 'results'
        self.verbose = verbose
    def insert_fold_records(self, model, config, random_state, fold, epoch, test_result, test_type):
        if model not in self.result:
            self.result[model] = {}
        if fold not in self.result[model]:
            self.result[model][fold] = {}
        if test_type not in self.result[model][fold]:
            self.result[model][fold][test_type] = FoldResult()
        fold_result = self.result[model][fold][test_type]

        f1, summary_df, _ = test_result.calc_results()
        if f1 > fold_result.best_f1_result:
            fold_result.best_f1_result = f1
            fold_result.best_f1_result_epoch = epoch
            fold_result.best_test_results = test_result
            fold_result.best_result_summary_df = summary_df

        if self.verbose:
            print(f'model: {model} | random_state: {random_state} | hidden_size: {config.hidden_size} | num_hidden_layers {config.num_hidden_layers} | num_attention_heads {config.num_attention_heads}')
            print(f'fold: {fold} | epoch: {epoch} | strategy: one_by_one')
            print(f'best_f1: {fold_result.best_f1_result:.3f} (epoch = {fold_result.best_f1_result_epoch})')
            print(summary_df)

    def print_best_results(self):
        print()
        print("---------------------------------------------------")
        for test_type in self.test_types:
            for model, all_data in self.result.items():
                for run, run_data in all_data.items():
                    if test_type in run_data:
                        fold_result = run_data[test_type]
                        print(f'#{run} {test_type:<11}: {model:<15} = best_f1: {fold_result.best_f1_result:.3f}')
        print("---------------------------------------------------")
        print()
    def save_fold_result(self, model, config, dataset, fold, date, expected_size, mlm_probability, random_state, train_size, test_size):
        for test_type in self.test_types:
            if model in self.result and fold in self.result[model] and test_type in self.result[model][fold]:
                fold_result = self.result[model][fold][test_type]
                fold_test_results = fold_result.best_test_results
                file_path = f'{self.result_dir}/{model}_{dataset}_{expected_size}_h{config.hidden_size}_l_{config.num_hidden_layers}_head{config.num_attention_heads}_{mlm_probability}_results.csv'
                y_true = list(map(lambda x: x.y_true, fold_test_results.results))
                y_pred = list(map(lambda x: x.y_pred, fold_test_results.results))
                model_name = f'{model}_{expected_size}_{mlm_probability}_{test_type}'
                assert(len(y_pred) == len(y_true))
                results = [{
                    "expected": y_true[i],
                    "predict": y_pred[i],
                    } for i in range(len(y_true))]
                with open(file_path, 'a') as f:
                    f.write(f'{dataset}|{model_name}|{train_size}|{test_size}|{fold}|{date}|{random_state}|{results}\n')

class TestResult:
    def __init__(self, y_true, y_pred):
        self.fold = -1
        self.y_true = y_true
        self.y_pred = y_pred
    def set_fold(self, fold):
        self.fold = fold
    def __str__(self):
        return f'fold-{self.fold}: {self.y_true} | {self.y_pred}'
class TestResults:
    def __init__(self):
        self.results = []
    def merge_result(self, test_result):
        self.results.extend(test_result.results)
    def size(self):
        return len(self.results)
    def add_results(self, results):
        self.results.extend(results)
    def calc_results(self):
        data = {
            'y_true': list(map(lambda x: x.y_true, self.results)),
            'y_pred': list(map(lambda x: x.y_pred, self.results)),
        }
        df = pd.DataFrame.from_dict(data)
        data_only_df = pd.DataFrame.from_dict(data)
        df['length'] = df['y_true'].apply(lambda result: len(result))
        df['true_f1_scores'] = df.apply(lambda result: true_f1(result.y_true, result.y_pred), axis=1)
        df['true_pairs_f1_scores'] = df.apply(lambda result: true_pairs_f1(result.y_true, result.y_pred), axis=1)
        df['bleu_scores'] = df.apply(lambda result: bleu(result.y_true, result.y_pred), axis=1)
        df['incorrect_f1_scores'] = df.apply(lambda result: calc_F1(result.y_true, result.y_pred), axis=1)
        df['incorrect_pairs_f1_scores'] = df.apply(lambda result: calc_pairsF1(result.y_true, result.y_pred), axis=1)
        summary_df = df.groupby(by=["length"]).agg({'length': 'size', 'true_f1_scores': 'mean', 'true_pairs_f1_scores': 'mean', 'bleu_scores': 'mean', 'incorrect_f1_scores': 'mean', 'incorrect_pairs_f1_scores': 'mean'})
        summary_df['count'] = summary_df['length']
        summary_df['length'] = summary_df.index
        summary_df = summary_df.set_index(np.arange(len(summary_df.index)))
        all_df = df.mean().to_frame().T
        all_df['length'] = f'{df["length"].mean():.2f}'
        all_df['count'] = summary_df['count'].sum()
        df = pd.concat([all_df, summary_df])
        f1 = df.head(1)['true_f1_scores'][0]
        #print(data_only_df)
        return f1, df, data_only_df

@dataclass
class FoldResult:
    best_f1_result: np.float64 = -1
    best_f1_result_epoch: int = 0
    best_test_results: TestResults = None
    best_result_summary_df: pd.core.frame.DataFrame = None


class Tester:
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.poi_tokenizer = BertTokenizer(vocab_file = config.poi_vocab_file, do_lower_case = False, do_basic_tokenize = False)
        self.start_predict_pos = 2 + self.config.num_extra_tokens
        self.max_length = self.config.max_position_embeddings + self.config.num_extra_tokens
        self.feature_size = 1
        print(f'Model: {self.config.model_type}, add_user_token:{self.config.add_user_token}, add_time_token: {self.config.add_time_token}, start_predict_pos: {self.start_predict_pos}, num_extra_tokens: {self.config.num_extra_tokens}')

    def run(self, filepath, model, epoch = 0, batch_size = 512, strategy = 'one_by_one', verbose = False):
        model = model.to(self.device).eval()
        results = []
        with open(filepath) as f:
            lines = f.readlines()
            test_batch_pois = np.empty((batch_size, self.max_length))
            #test_batch_timestamps = np.empty((batch_size, 2))
            i = 0
            cnt = 0
            size = len(lines)
            with torch.no_grad():
                for line in lines:
                    pois = self.format_line(model.config, line)
                    test_batch_pois[i, :] = pois
                    #test_batch_timestamps[i, :] = timestamps
                    i += 1
                    if i % batch_size == 0:
                        results = results + self.evaluate_batch(model, test_batch_pois, strategy)
                        cnt += i
                        print(f'Progress: {cnt} / {size}')
                        i = 0

                if i > 0:
                    test_batch_pois = test_batch_pois[:i]
                    cnt += i
                    results = results + self.evaluate_batch(model, test_batch_pois, strategy)
        testResults = TestResults()
        testResults.add_results(results)
        return testResults
    def evaluate_batch(self, model, test_batch_pois, strategy):
        config = model.config
        mask_token_id = 0
        cls_token_id = 1
        pad_token_id = 2
        sep_token_id = 3
        unk_token_id = 4
        poi_ids = torch.tensor(test_batch_pois, dtype = torch.int64, device = self.device)

        attention_mask = torch.full(poi_ids.shape, 1).to(self.device)
        attention_mask[(poi_ids == pad_token_id)] = 0
        expected_poi_ids = poi_ids.clone()
        max_sequence_length = poi_ids.shape[1]
        input_size = torch.full((1, expected_poi_ids.shape[0]), 0).to(self.device)
        result = torch.where(poi_ids == sep_token_id)
        expected_outputs = {'input_ids': poi_ids.clone()}
        result = list(zip(result[0], result[1]))
        for row, column in result:
            # [CLS] USER TIME_1 TIME_2 POI_1 MASK
            poi_ids[row, self.start_predict_pos: column - 1] = mask_token_id
        torch.set_printoptions(threshold=10_000,edgeitems= 100000)
        max_trajectory_length = max_sequence_length - 2
        inputs = {'input_ids': poi_ids, 'attention_mask': attention_mask}
        if strategy in ['one_by_one', 'all_in_once']:
            if strategy == 'one_by_one':
                predict_poi_ids = self.one_by_one(model, inputs, expected_outputs, max_trajectory_length, mask_token_id, unk_token_id)
                stacks = torch.stack([expected_poi_ids, predict_poi_ids], dim = 1)
            elif strategy == 'all_in_once':
                predict_poi_ids = self.all_in_once(model, inputs, expected_outputs, max_trajectory_length, mask_token_id, unk_token_id)
                stacks = torch.stack([expected_poi_ids, predict_poi_ids], dim = 1)
            results = []
            for tuple in stacks:
                y_true = tuple[0]
                y_pred = tuple[1]
                start_index = self.start_predict_pos - 1
                end_index = (y_true == sep_token_id).nonzero()[0]
                y_true = self.poi_tokenizer.convert_ids_to_tokens(y_true[start_index:end_index])
                y_pred = self.poi_tokenizer.convert_ids_to_tokens(y_pred[start_index:end_index])
                if len(y_true) < 3 or len(y_pred) < 3:
                    print(y_true, y_pred)
                results.append(TestResult(y_true, y_pred))

        return results
    def all_in_once(self, model, inputs, expected_outputs, max_trajectory_length, mask_token_id, unk_token_id):
        poi_ids = inputs['input_ids']
        masked = (poi_ids == mask_token_id)
        inputs['input_ids'][masked] = mask_token_id
        output = model(**inputs)
        logits = output.logits
        indices = torch.argmax(logits, dim = -1)
        inputs['input_ids'][masked] = indices[masked]
        return inputs['input_ids']

    def one_by_one(self, model, inputs, expected_outputs, max_trajectory_length, mask_token_id, unk_token_id):
        poi_ids = inputs['input_ids']
        expected = expected_outputs['input_ids']
        for masked_index in range(self.start_predict_pos, max_trajectory_length):
            masked = torch.full(poi_ids.shape, False)
            masked[:, masked_index] = True
            masked2 = torch.full(poi_ids.shape, False)
            masked2[(poi_ids == mask_token_id)] = True
            masked = masked & masked2
            if len((masked2 == True).nonzero()) == 0:
                break
            output = model(**inputs)
            logits = output.logits
            indices = torch.argmax(logits, dim = -1)
            inputs['input_ids'][masked] = indices[masked]
        return inputs['input_ids']

    def format_line(self, config, line):
        input_ids = []
        sep = ' '
        poi_ids = np.full((self.max_length), self.config.pad_token_id)
        features = line.strip().split(self.config.feature_sep)

        user_id = features[0]
        if self.config.add_user_token:
            input_ids.append(user_id)
        if self.config.add_time_token:
            times = [datetime.fromtimestamp(int(i)) for i in features[-1].split(self.config.time_series_sep)]
            times = [datetime_to_interval(t) for t in times]
            times = [f'time-{_}' for _ in times]
            times = [times[0], times[-1]]
            times = ' '.join(times)
            input_ids.append(times)

        trajectory = features[1].replace(self.config.time_series_sep, sep)
        input_ids.append(trajectory)

        input_ids = f'[CLS] {sep.join(input_ids)} [SEP]'
        input_ids = self.poi_tokenizer.tokenize(input_ids)
        input_ids = self.poi_tokenizer.convert_tokens_to_ids(input_ids)
        size = len(input_ids)
        poi_ids[:size] = input_ids
        return poi_ids
