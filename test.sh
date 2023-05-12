cd megatron/data; make

# Prepare the test data
cd ../../
python prepare_data.py -d ./data --tokenizer HFTokenizer --vocab-file /fsx/pile/20B_tokenizer.json

# Run the test
python ./deepy.py train.py ./configs/test_config