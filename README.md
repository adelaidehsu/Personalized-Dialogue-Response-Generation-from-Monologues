# Personalized Dialogue Response Generation from Monologues
This is the code for the paper [Personalized Dialogue Response Generation from Monologues](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1696.html) by Feng-Guang Su*, Aliyah R. Hsu*, Yi-Lin Tuan and Hung-Yi Lee (* indicates equal contribution). See the [project page](https://adelaidehsu.github.io/Personalized-Dialogue-Response-Generation-learned-from-Monologues-demo/) for details and generated sentence examples.

## Training
Please configure train.sh first with your own settings. (e.g. model path and paprameters, data path, feature path and glove model path)
Once train.sh is complete, run the following:
```
bash train.sh
```

## Testing
For single sentence testing (1 prompt -> 1 response from your desired persona), please run:
```
bash test.sh
```
For multiple sentence testing (1 prompt -> multiple responses from different personas), please run:
```
bash seriestest.sh
```
For automatic testing by generating sentence files, please run:
```
bash filetest.sh
```
## Note
If you are using your own dataset, please refer to our provided data directory for the correct format of conversations, monologues and features.

## Citation
If you use any of our code in your work, please cite:
```bash
@inproceedings{Su2019PersonalizedDR,
  title={Personalized Dialogue Response Generation Learned from Monologues},
  author={Feng-Guang Su and Aliyah R. Hsu and Yi-Lin Tuan and Hung-yi Lee},
  booktitle={Interspeech},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:202740149}
}
```
