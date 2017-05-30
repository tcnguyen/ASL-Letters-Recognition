import os

for p in people:
    os.mkdir("./datasets/dataset5_120/"+p)
    for c in letters:
        os.mkdir("./datasets/dataset5_120/"+p+"/"+c)

for p in people:
    for c in letters:
        for img_path in dataset[p][c]:
            new_path = img_path.replace("dataset5", "dataset5_120")
            with Image.open(img_path) as img:
                new_img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
                new_img.save(new_path)
