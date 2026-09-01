---
license: cc-by-nc-4.0
dataset_info:
  features:
  - name: file
    dtype: string
  - name: audio
    struct:
    - name: array
      sequence: float64
    - name: sampling_rate
      dtype: int64
  - name: label
    dtype: string
  - name: utt_type
    dtype: string
  splits:
  - name: train
    num_bytes: 215847527002
    num_examples: 220963
  download_size: 69218972580
  dataset_size: 215847527002
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- vi
tags:
- speaker-verification
- speaker-recognition
- spoofing-aware-speaker-verification
- vietnamese
pretty_name: VSASV Dataset
size_categories:
- 100K<n<1M
---
This paper introduces the VSASV dataset for Spoofing-Aware Speaker Verification (SASV) in Vietnamese language. A part of this dataset was used to host the Spoofing-aware Speaker Verification Challenge in [VLSP 2023](https://vlsp.org.vn/vlsp2023/eval/vsasv).
Due to some fault in data storage, we are sorry that the published data is not as complete as described in the paper. However, we are working on an extended version of the dataset with Vietnamese TTS audios included within 2025. Stay tuned!


The paper has been accepted at [INTERSPEECH 2024](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html). Please cite as:
```
@inproceedings{hoang24b_interspeech,
  title     = {VSASV: a Vietnamese Dataset for Spoofing-Aware Speaker Verification},
  author    = {Vu Hoang and Viet Thanh Pham and Hoa Nguyen Xuan and Pham Nhi and Phuong Dat and Thi Thu Trang Nguyen},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4288--4292},
  doi       = {10.21437/Interspeech.2024-1972},
  issn      = {2958-1796},
}
```
