# Audio Style Transfer using DeepAFx-ST

## 참고 논문
[Style transfer of audio effects with differentiable signal processing](https://arxiv.org/pdf/2207.08759.pdf)

## 환경 세팅
```shell
conda create -n deepafx-st python=3.8 -y
conda activate deepafx-st

# Update pip and install
pip install --upgrade pip
pip install --pre -e .

# Linux install
apt-get install libsndfile1
apt-get install sox
apt-get install ffmpeg
apt-get install wget
```


## 코드 실행 방법
```python
python process.py -i [INPUT_AUDIO] -r [REFERENCE_AUDIO] -c model.ckpt
```