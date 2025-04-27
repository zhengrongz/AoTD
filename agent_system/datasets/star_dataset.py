import json
from torch.utils.data import Dataset


class STARDataset(Dataset):
    def __init__(self, data_dir):
        self.qa_datas = json.load(open(data_dir))

    def __len__(self):
        return len(self.qa_datas)
    
    def __getitem__(self, idx):

        qa_data = self.qa_datas[idx]
        question_id = qa_data['question_id']
        question = qa_data['question']
        video_id = qa_data['video_id']
        
        answer = qa_data['answer']
        possible_answers = [choice['choice'] for choice in qa_data['choices']]
        answer = chr(65 + possible_answers.index(answer))

        return {
            "question_id": question_id,
            "question": question,
            "video_id": video_id,
            "possible_answers": possible_answers,
            "answer": answer
        }