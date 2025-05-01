## Distillation
This part provides a distillation framework for AoTD. There are two steps for training:

1. Customize your dataset.
2. Modify the training script.

## Dataset Configuration
The core of the dataset is the suffix prompt:

### Example Instances:

1. **Answering data**:
```json
{
    "video_path": "videos/001.mp4", 
    "conversation": [
        {"role": "user", 
         "content": [{"type": "text", "text": "Question: how many children are in the video\nOptions:\n(A) one\n(B) three\n(C) seven\n(D) two\n(E) five\nAnswer with the option's letter from the given choices directly and only give the best option."}, {"type": "video"}]
         }, 
         {"role": "assistant", "content": [{"type": "text", "text": "D"}]}]
}
```

2. **Reasoning data**:
```json
{
    "video_path": "videos/001.mp4",
    "conversations": [
        {"role": "user", 
          "content": [{"type": "text", "text": "Question: how many children are in the video\nOptions:\n(A) one\n(B) three\n(C) seven\n(D) two\n(E) five\nExplain the rationale to answer the question."}, {"type": "video"}]
          }, 
          {"role": "assistant", "content": [{"type": "text", "text": "To solve this question, we have to find the maximum number of children appearing in the video at the same time ..."}]
          }
    ]
}
```

Different model may require different data format, but the suffix prompt should remain unchanged.

After you customized your dataset, you need to add it in the `datasets/train_datasets.yaml`.

As we do not set extra `video_folder_path`, the `video_path` in the data leads to the video should be the absolute path.

## Training
You can directly run the scripts to train the model.
```
sh train.sh
```

The configuration about the trainer is in the `train.py`, and you can modify it. (To be optimized.)