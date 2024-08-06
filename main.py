import train
import test
import data_loader

def main():
    audios_numpy, labels = data_loader.get_sampels()
    train.main(audios_numpy, labels)  # TODO https://github.com/Xtra-Computing/thundersvm/tree/master/python use with GPU
    test.main()

if __name__ == '__main__':
    main()