import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from svm import train as svm_train, test as svm_test
from linear_regression import train as linear_train, test as linear_test
from knn import run as knn_run
from logistic import run as logistic_run
from manual_ann_regression import run as manual_ann_run
from manual_ann_classification import run as manual_ann_classification_run
from cnn_cifar10 import run as cnn_cifar10_run

def main():
    if len(sys.argv) != 3:
        print("用法：python main.py [algo] [mode]")
        print("例如：python main.py svm train")
        print("例如：python main.py svm test")
        print("例如：python main.py linear train")
        print("例如：python main.py linear test")
        print("例如：python main.py knn run")
        print("例如：python main.py logistic run")
        return

    algo = sys.argv[1].lower()
    mode = sys.argv[2].lower()

    if algo == "svm":
        if mode == "train":
            svm_train()
        elif mode == "test":
            svm_test()
        else:
            print("svm 只支持 train / test")

    elif algo == "linear":
        if mode == "train":
            linear_train()
        elif mode == "test":
            linear_test()
        else:
            print("linear 只支持 train / test")

    elif algo == "knn":
        if mode == "run":
            knn_run()
        else:
            print("knn 目前只支持 run")

    elif algo == "logistic":
        if mode == "run":
            logistic_run()
        else:
            print("logistic 目前只支持 run")
    elif algo == "manual_ann":
        if mode == "run":
            manual_ann_run()
        else:
            print("manual_ann 只支持 run")

    else:
        print("不支持的算法")

if __name__ == "__main__":
    main()
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ann_house import run as ann_house_run
from ann_titanic import train as ann_titanic_train, test as ann_titanic_test, run as ann_titanic_run
from ann_cifar10 import run as ann_cifar10_run


def main():
    if len(sys.argv) != 3:
        print("用法：python main.py [任务] [模式]")
        print("例如：python main.py ann_house run")
        print("例如：python main.py ann_titanic train")
        print("例如：python main.py ann_titanic test")
        print("例如：python main.py ann_cifar10 run")
        return

    task = sys.argv[1].lower()
    mode = sys.argv[2].lower()

    if task == "ann_house":
        if mode == "run":
            ann_house_run()
        else:
            print("ann_house 只支持 run")

    elif task == "ann_titanic":
        if mode == "train":
            ann_titanic_train()
        elif mode == "test":
            ann_titanic_test()
        elif mode == "run":
            ann_titanic_run()
        else:
            print("ann_titanic 支持 train / test / run")

    elif task == "ann_cifar10":
        if mode == "run":
            ann_cifar10_run()
        else:
            print("ann_cifar10 只支持 run")
    
    elif algo == "ann_classification":
        if mode == "run":
            manual_ann_classification_run()
        else:
            print("ann_classification 只支持 run")

    elif algo == "cnn_cifar10":
        if mode == "run":
            cnn_cifar10_run()
        else:
            print("cnn_cifar10 只支持 run")

    else:
        print("不支持的任务")


if __name__ == "__main__":
    main()