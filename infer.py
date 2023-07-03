import argparse
import functools
import warnings

from src.predictor import Tacotron2Predictor
# from src.utils.utils import add_arguments, print_arguments
warnings.filterwarnings('ignore')


def main():
    # parser = argparse.ArgumentParser(description=__doc__)
    # add_arg = functools.partial(add_arguments, argparser=parser)
    # add_arg('configs',    str,  'configs/config.yml',       "配置文件")
    # add_arg('use_gpu',    bool, False,                         "是否使用GPU预测")
    # add_arg('model_path', str,  'models/Tacotron2/best_model', "预测模型文件路径")
    # add_arg('enhance',    bool, True,                          "对生成的语音是否去噪")
    # args = parser.parse_args()
    # print_arguments(args=args)

    predictor = Tacotron2Predictor()
    text = 'Hello World'
    out_path = './001.wav'
    predictor.predict(line=text, path=out_path)


if __name__ == "__main__":
    main()
