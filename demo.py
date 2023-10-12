import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = '/home/apujol/project/datasets/coco/train2017',
                        annFile = '/home/apujol/project/datasets/coco/annotations/captions_train2017.json',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target)

t = transforms.ToPILImage()
img = t(img)
img.save("out.jpg")