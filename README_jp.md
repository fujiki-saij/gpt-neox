# JP StableLM
## Install
Please follow [GPT-NeoX Setup Guide for Stable-LM](https://docs.google.com/document/d/1vV4fPgv5OkBtoTqAR86f3i4_xWIbnJN3td38g8m6ieY/edit).

But the only difference is that you need to install our custom Japanese lm-evaluation-harness. 
After all installation is done, install `lm_eval` as follows:
```
git clone -b jp-stable https://github.com/Stability-AI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```
