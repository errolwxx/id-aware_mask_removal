import os
import numpy as np
import random
import shutil

# images_path = "./datasets/pins_celebs/images"
# masks_path = "./datasets/pins_celebs/masks"

# allimages = [os.path.join(images_path, f) for _, _, files in os.walk(images_path) for f in files]
# allmasks = [os.path.join(masks_path, f) for _, _, files in os.walk(masks_path) for f in files]

root = "../Lightweight-Face-Recognition/ResNet/datasets"

images_path = "./datasets/pins_celebs/originals"
allimages = [os.path.join(images_path, f) for _, _, files in os.walk(images_path) for f in files]

names = ['Mark_Zuckerberg',
 'Jennifer_Lawrence',
 'Bobby_Morley',
 'Adriana_Lima',
 'Jason_Momoa',
 'Pedro_Alonso',
 'Megan_Fox',
 'Logan_Lerman',
 'Chris_Pratt',
 'Avril_Lavigne',
 'Leonardo_DiCaprio',
 'Emilia_Clarke',
 'Sophie_Turner',
 'Store',
 'scarlett_johansson',
 'Miley_Cyrus',
 'Morena_Baccarin',
 'Brie_Larson',
 'Jeremy_Renner',
 'Morgan_Freeman',
 'Rami_Malek',
 'Emma_Stone',
 'alycia_dabnem_carey',
 'Taylor_Swift',
 'Alvaro_Morte',
 'Zoe_Saldana',
 'Richard_Harmon',
 'Elizabeth_Lail',
 'Brian_J._Smith',
 'Alex_Lawther',
 'Rihanna',
 'Marie_Avgeropoulos',
 'Alexandra_Daddario',
 'Eliza_Taylor',
 'amber_heard',
 'Tom_Holland',
 'Bill_Gates',
 'Rebecca_Ferguson',
 'Madelaine_Petsch',
 'Irina_Shayk',
 'Robert_Downey_Jr',
 'Ben_Affleck',
 'Keanu_Reeves',
 'elon_musk',
 'Lili_Reinhart',
 'Millie_Bobby_Brown',
 'Chris_Hemsworth',
 'Christian_Bale',
 'Dominic_Purcell',
 'Tom_Cruise',
 'Andy_Samberg',
 'Zac_Efron',
 'Shakira_Isabel_Mebarak',
 'jeff_bezos',
 'Natalie_Dormer',
 'Wentworth_Miller',
 'barack_obama',
 'Ursula_Corbero',
 'barbara_palvin',
 'kiernen_shipka',
 'Sarah_Wayne_Callies',
 'grant_gustin',
 'Jake_Mcdorman',
 'Hugh_Jackman',
 'Krysten_Ritter',
 'Tom_Hardy',
 'Josh_Radnor',
 'Jimmy_Fallon',
 'Henry_Cavil',
 'Danielle_Panabaker',
 'tom_ellis',
 'Mark_Ruffalo',
 'Katharine_Mcphee',
 'Gwyneth_Paltrow',
 'Anthony_Mackie',
 'Lionel_Messi',
 'Cristiano_Ronaldo',
 'gal_gadot',
 'Katherine_Langford',
 'Tom_Hiddleston',
 'Jessica_Barden',
 'Johnny_Depp',
 'elizabeth_olsen',
 'Chris_Evans',
 'Penn_Badgley',
 'Emma_Watson',
 'Neil_Patrick_Harris',
 'margot_robbie',
 'Maisie_Williams',
 'Lindsey_Morgan',
 'Dwayne_Johnson',
 'camila_mendes',
 'Stephen_Amell',
 'Zendaya',
 'Anne_Hathaway',
 'Maria_Pedraza',
 'Selena_Gomez',
 'Inbar_Lavi',
 'Amanda_Crew',
 'Robert_De_Niro',
 'Nadia_Hilker',
 'Brenton_Thwaites',
 'ellen_page',
 'Natalie_Portman',
 'melissa_fumero',
 'Tuppence_Middleton']

name_dict = {}

for name in names:
    name_dict[name] = []

for image in allimages:
    for name in names:
        if image.split("/")[-1][:-9] == name:
            name_dict[name].append(image)

for k, v in name_dict.items():
    name_path = os.path.join(root, k)
    if not os.path.exists(name_path):
        os.mkdir(name_path)
    for i in v:
        shutil.copy(i, name_path)
