import json
from torch.utils.data import Dataset
import csv
import codecs

class NEXTDataset(Dataset):
    def __init__(self, data_dir):
        self.qa_datas = []
        with codecs.open(data_dir, encoding='utf-8-sig') as f:
            for i,row in enumerate(csv.DictReader(f, skipinitialspace=True)):
                self.qa_datas.append(row)
        f.close()

    def __len__(self):
        return len(self.qa_datas)
    
    def __getitem__(self, idx):

        qa_data = self.qa_datas[idx]

        question = qa_data['question']
        video_id = qa_data['video']
        
        answer = qa_data['answer']
        possible_answers = [qa_data['a0'], qa_data['a1'], qa_data['a2'], qa_data['a3'], qa_data['a4']]
        answer = chr(65 + int(answer))

        return question, video_id, possible_answers, answer