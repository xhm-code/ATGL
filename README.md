# ATGL

Code for the ACL 2026 main conference paper [ATGL: An Adaptive-Threshold Global Loss for Document-level Relation Extraction](https://aclanthology.org/2026.acl-long.1603/).

**If you find our model, code, or paper helpful, please consider citing our work 📝 and starring our repository ⭐️!**

```bibtex
@inproceedings{xu2026atgl,
  title={ATGL: An Adaptive-Threshold Global Loss for Document-level Relation Extraction},
  author={Xu, Huangming and Zhang, Fu and Yang, Zhixuan and Zhang, Lu and Cheng, Jingwei},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={34702--34716},
  year={2026}
}
```
---
## Requirements
Ensure you have the following packages installed:

* Python (tested on 3.7.4)
* CUDA (tested on 11.6)
* [PyTorch](http://pytorch.org/) (tested on 1.12.0+cu113)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.20.1)
* numpy (tested on 1.21.6)
* [spacy](https://spacy.io/) (tested on 3.3.3)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm
* wandb

---
## Dataset
Go to the `dataset` folder and extract the `redocred.tar.gz` compressed file. 
Organize your dataset files as follows:
```
ATGL
 |-- dataset
 |    |-- redocred
 |    |    |-- train_revised.json        
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |    |-- train_distant.json  
 |-- meta
 |   |-- rel2id.json
 |-- scripts
 |-- checkpoint
 |-- result
```

---
## Training and Evaluation

Use the following commands to train and evaluate the model.

#### On ReDocRED Dataset 
- **Using RoBERTa**:
    ```bash
    # Training
    bash scripts/run_roberta_redocred_seeds.sh
    
    # Evaluation
    bash scripts/test_roberta_redocred_seeds.sh
    ```
  
- **Using BERT**:
    ```bash
    # Training
    bash scripts/run_bert_redocred_seeds.sh
    
    # Evaluation
    bash scripts/test_bert_redocred_seeds.sh
    ```
---

Note: This code is partially based on the code of [ATLOP](https://github.com/wzhouad/ATLOP) and [HingeABL](https://github.com/Jize-W/HingeABL).
