# Yet another pre-trained model for Thai BERT
#### _thbert_

BERT, a pre-trained unsupervised natural language processing model, prepared for fine-tuning to perform NLP downstream tasks significantly.

To enable research oppotunities with very few Thai Computational Linguitic resources, we willingly introduce fundamental language resouces, Thai BERT, build from scratch for researchers and enthusiast. 

## Pre-trained models

* **[`THBERT-Base, uncased`](https://1drv.ms/u/s!AkQdzVQXmrE5goIhhk7kCCzs13EwQg?e=YLSwSF)**: Thai, 12-layer, 768-hidden, 12-heads
* **[`THBERT-Large, uncased`](https://1drv.ms/u/s!AkQdzVQXmrE5goIjtVsLCiMB08SuNA?e=yxLszV)**: Thai, 24-layer, 1024-hidden, 16-heads

Each .zip file contains three items:
*   A TensorFlow checkpoint (`thbert_model.ckpt`) containing the pre-trained weights (3 files).
*   A vocab file (`vocab.txt`) to map WordPiece to word id.
*   A config file (`bert_config.json`) which specifies the hyperparameters of the model.

### Pre-training data
#### Source
* **[`thwiki_dump`](https://dumps.wikimedia.org/thwiki/)**
  * https://dumps.wikimedia.org/thwiki/20200401/
  * More than 800K sentences/paragraphs
* **[`THAI-NEST`](https://saki.siit.tu.ac.th/kindml/thainest/index.php/download)**: Soon
  * https://saki.siit.tu.ac.th/kindml/thainest/download/Thai_Nest_a_framework_TheeramunkongT_2010.pdf
* **[`BEST2010`](http://thailang.nectec.or.th/best/)**: Soon
* **[`ORCHID`](http://www.hlt.nectec.or.th/orchid/)**: Soon


#### Tokenization
* **[`sentencepiece`](https://github.com/google/sentencepiece)**
  * unigram
  * 128K vocabulary size
