# TIE: Topological Information Enhanced Structural Reading Comprehension on Web Pages

**T**opological **I**nformation **E**nhanced (TIE) model leverages the informative topological structures of the web
pages to tackle the web base **S**tructure **R**eading **C**omprehension (SRC) task, and achieves the SOTA results on
[WebSRC dataset](https://x-lance.github.io/WebSRC/) at the time of writing. This repository is the full implementation
of our TIE model. For more details, please refer to our paper:

[TIE: Topological Information Enhanced Structural Reading Comprehension on Web Pages](https://arxiv.org/abs/2205.06435)

## Requirements

The required python packages is listed in "requirements.txt". You can install them by
```commandline
pip install -r requirements.txt
```
or
```commandline
conda install --file requirements.txt
```

## Data preparing

First, please following the data pre-processing guidelines in [the WebSRC office repository](https://github.com/X-LANCE/WebSRC-Baseline#data-pre-processing).
Then, in order to form the NPR graph efficiently afterwards, we calculate and store the NPR relations between valid tags
of each web page in a dictionary format. To achieve this, run
```commandline
python src/data_preprocess.py --root_dir ./data --task rect_mask
```
The resulting dictionary for each web page will be placed in the same directory as the corresponding html file while the
name of the resulting file has an additional suffix `.relation.json`

## Training

After completing the data preparing steps, TIE can be trained by running the `train.sh` file in the folder
`script/{backbone-PLM-for-CE}`. As you can see, the backbone model used for the Content Encoder of TIE is specified in
the directory of the bash files. For example, to train TIE with MarkupLM as its Content Encoder, run
```commandline
bash ./script/MarkupLM/train.sh
```
Moreover, to reproduce the experiments in ablation study, you can use the argument `--mask` to specify the GAT masks
used in TIE and the argument `--direction` to specify the relations used in NPR graph.

## Evaluation

Similarly, the bash file for evaluation can be found in the same directory as the bash file for training. Specifically,
the corresponding `eval_stage_1.sh` file evaluates the quality of TIE's tag predictions all the saved checkpoints on the
development set, while `eval_stage_2.sh` file evaluates the final answer span predictions on the development set where
an additional token-level QA model with its model type and a checkpoint of TIE need to be specified. For example, to
evaluate the tag prediction quality of all the checkpoints saving by the previous example command, run
```commandline
bash ./script/MarkupLM/eval_stage_1.sh
```
Then, for answer refining stage, suppose that we use **MarkupLM** which is stored in folder `./token_QA` as the
additional token-level QA model and the checkpoint we want to evaluate is located at `./result/MarkupLM/checkpoint-27000`.
Note that the previous example command will store the n best answer tag prediction in a corresponding json file, in this
case, the json file will be `./result/MarkupLM/nbest_predictions_27000.json`. Therefore, to evaluate the final
performance, run
```commandline
bash ./script/MarkupLM/eval_stage_2.sh markuplm ./token_QA ./result/MarkupLM/nbest_prediction_27000.json
```

## Reference

If you find TIE useful or inspiring in your research, please cite the corresponding paper. The bibtex are listed below:
```text
@article{zhao-etal-2022-tie,
  author    = {Zihan Zhao and
               Lu Chen and
               Ruisheng Cao and
               Hongshen Xu and
               Xingyu Chen and
               Kai Yu},
  title     = {{TIE:} Topological Information Enhanced Structural Reading Comprehension
               on Web Pages},
  journal   = {CoRR},
  volume    = {abs/2205.06435},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.06435},
  doi       = {10.48550/arXiv.2205.06435},
  eprinttype = {arXiv},
  eprint    = {2205.06435}
}
```

## License

This project is licensed under the license found in the LICENSE file. Portions of the source code are based on the
official code of [WebSRC](https://github.com/X-LANCE/WebSRC-Baseline) and [MarkupLM](https://github.com/microsoft/unilm/tree/master/markuplm)
