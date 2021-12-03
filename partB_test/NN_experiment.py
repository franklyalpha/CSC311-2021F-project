import matplotlib.pyplot as plt
import json


def main():
    with open("../part_a/nn0_valid_acc.txt", "r") as fp:
        nn0_valid_acc = json.load(fp)
    with open("./nn2_epoch.txt", "r") as fp:
        epoch = json.load(fp)
    with open("./nn2_valid_acc.txt", "r") as fp:
        nn2_valid_acc = json.load(fp)
    with open("./nn4_valid_acc.txt", "r") as fp:
        nn4_valid_acc = json.load(fp)
    with open("./nn6_valid_acc.txt", "r") as fp:
        nn6_valid_acc = json.load(fp)

    nn0_max = nn0_valid_acc.index(max(nn0_valid_acc))
    nn2_max = nn2_valid_acc.index(max(nn2_valid_acc))
    nn4_max = nn4_valid_acc.index(max(nn4_valid_acc))
    nn6_max = nn6_valid_acc.index(max(nn6_valid_acc))

    plt.plot(epoch, nn0_valid_acc, "-cD", markevery=[nn0_max], label='0')
    plt.plot(epoch, nn2_valid_acc, "-bD", markevery=[nn2_max], label='2')
    plt.plot(epoch, nn4_valid_acc, "-gD", markevery=[nn4_max], label='4')
    plt.plot(epoch, nn6_valid_acc, "-rD", markevery=[nn6_max], label='6')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.legend()
    plt.title(
        "Neural Network with Different Number of Hidden Layers")
    plt.savefig("experiment.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()


