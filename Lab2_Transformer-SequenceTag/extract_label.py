with open('./data/train_TAG.txt', 'r') as file:
    lines = file.readlines()

label_set = set()

for line in lines:
    tags = line.strip().split()
    for tag in tags:
        label_set.add(tag)
        
with open('./data/extracted_labels.txt', 'w') as file:
    for label in label_set:
        file.write(label + '\n')