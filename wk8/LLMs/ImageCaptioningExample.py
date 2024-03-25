# Program extended from https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
# Look at that link for details about the model's neural architecture
# Dependencies:
# pip install transformers torch

import os
import random
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


path = r"flickr8k-resised"
for image in os.listdir(path):
    if random.random() < 0.003:
        fileName = os.path.join(path, image)
        caption = predict_step([fileName])
        line = "%s %s" % (image, caption)
        print(line)


# 3226541300_6c81711e8e.jpg ['a man that is standing up with a stick in his hand']
# kindda make sense
# 396360611_941e5849a3.jpg ['a person standing on top of a rock near a body of water']
# accurate
# 2676649969_482caed129.jpg ['a woman is sitting in front of a fence']
# accurate
# 3084731832_8e518e320d.jpg ['a dog that is standing in the snow']
# yes accurate
# 3200120942_59cfbb3437.jpg ['a pair of skis are attached to a ski lift']
# kinda, its people on the ski lift
# 2125216241_5b265a2fbc.jpg ['a large boat is docked at the dock']
# accurate
# 2657663775_bc98bf67ac.jpg ['a man running on the beach with a frisbee']
# accurate
# 2831217847_555b2f95ca.jpg ['a truck driving down a muddy road next to a forest']
# accurate
# 3271468462_701eb88d3b.jpg ['a brown and white horse laying next to a brown and white horse']
# not accurate, dog biting a horse
# 2402462857_7684848704.jpg ['a black and white dog standing in the middle of a field']
# road not a field
# 3498417123_3eae6bbde6.jpg ['a team of soccer players playing a game of soccer']
# rugby not soccer
# 3475552729_a3abd81ee6.jpg ['a little boy playing with a soccer ball']
# accurate
# 490044494_d2d546be8d.jpg ['a pair of dogs running through a grassy field']
# 3 dogs instead of 2
# 451597318_4f370b1339.jpg ['a dog jumping in the air to catch a frisbee']
# accurate
# 3451345621_fe470d4cf8.jpg ['a man standing next to a woman holding a sign']
# accurate
# 3131990048_369b081021.jpg ['a bird that is flying over a grassy area']
# accurate but the bird is trying to catch a mice
# 3646927481_5e0af1efab.jpg ['motorcycles are parked in a field']
# accurate, but they are doing some sort of event
# 2635908229_b9fc90d3fb.jpg ['a young boy standing on top of a wooden pole']
# sitting, not standing
# 3621623690_0095e330bc.jpg ['a person jumping a horse over a fence']
# not fence, show jumping thing
# 3564157681_03a13b7112.jpg ['a dog jumping up to catch a frisbee']
# accurate
# 3103340819_46de7954a9.jpg ['a man sitting on the ground next to a door']
# also a woman,
# 2427490900_5b7a8874b9.jpg ['a large group of people sitting in front of a building']
# accurate, building is a monument
# 3437147889_4cf26dd525.jpg ['a man riding a dirt bike on top of a dirt field']
# accurate
# 3226254560_2f8ac147ea.jpg ['a cow walking across a snow covered field']
# accurate, the dog is shitting
# 56494233_1824005879.jpg ['a person standing on top of a snow covered mountain']
# accurate
# 339822505_be3ccbb71f.jpg ['a man and a woman looking at a bunch of bananas']
# not accurate, this is a carnival dress
# 1211015912_9f3ee3a995.jpg ['a young girl holding a pink frisbee in a park']
# not accurate, a bunch children, some holding balloons.
# 3417231408_6ce951c011.jpg ['a man cooking food on top of a fire hydrant']
# not accurate, its a pizza oven

# 12 accurate of out 28 captions, 7 are semi-accurate, 9 are not accurate.
