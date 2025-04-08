import argparse
import torch
import fmodel
import json

parser = argparse.ArgumentParser(description='Parser for predict.py')

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

def main():
    model = fmodel.load_checkpoint(path)
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)

    probabilities = fmodel.predict(path_image, model, number_of_outputs, device)
    probability = probabilities[0][0]
    labels = [name[str(index + 1)] for index in probabilities[1][0]]

    for i in range(len(labels)):
        print("{} with a probability of {}".format(labels[i], probability[i]))
    print("Finished Predicting!")

if __name__ == "__main__":
    main()
