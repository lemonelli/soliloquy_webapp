# Soliloquy Webapp

 - Deploys a web application which demonstrates the paraphrase augmentation algorithm developed in concert with [Soliloquy Variation](https://github.com/ucdaviscl/soliloquy_variation) and [Soliloquy 2019](https://github.com/ssdavidson/soliloquy_2019)
 - Takes in a training file, creates paraphrases of training data, allows user to select paraphrases to add to the training data, retrains and evaluates the model.
 
## Dependencies
- Python >= 3.6
- [PyFST](https://github.com/placebokkk/pyfst) with OpenFST (required for fst sentence variation)
- [OpenNMT](https://github.com/OpenNMT/OpenNMT) with GPU support (required for neural language model)
- [KenLM](https://github.com/kpu/kenlm) and its Python module (required for paraphrase testing)
- [Soliloquy Variation](https://github.com/ucdaviscl/soliloquy_variation) (required for paraphrase generation)
- [Soliloquy 2019](https://github.com/ssdavidson/soliloquy_2019) (required for language model evaluation)
- Python Packages in 'requirements.txt'
## External Data

Models and data are available here (password required):
https://ucdavis.box.com/v/soliloquy 

## Deployment

This webapp is deployed using a docker container built from an image file.  The dockerhub image file can be found [here](https://cloud.docker.com/u/lemonelli/repository/docker/lemonelli/soliloquy-webapp) under the tag 'aug-model'.  This container alreaedy has all dependencies installed and can be initialized and run on any machine running docker.
