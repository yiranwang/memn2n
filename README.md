# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) using Python and Keras in the backend.  
Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.  

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

This is a fork of [priyank87's repository](https://github.com/priyank87/memn2n)  

## How to use  

`git clone <this repository>`  

`cd memn2n`  

`mkdir server/model/babi_memnn_1_1` -> specify where you would store the trained model  

`python server/babi_memnn_training.py` -> training  

__you can specify the model/output path/training epochs/challenge type in server/config.py__  
__please refer to the code or contact me directly for further clarification.__  

`python app.py` -> serve up the web app  

Now go to localhost:5000 in your browser.
