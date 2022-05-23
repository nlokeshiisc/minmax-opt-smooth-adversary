#encoding: utf-8
'''
Create a font dataset:
Content / size / color(Font) / color(background) / style
E.g. A / 64/ red / blue / arial
'''
from imp import create_dynamic
import os
import pygame
import pickle as pkl
from PIL import Image
from sqlalchemy import Integer
import numpy as np
from tqdm import tqdm

# Added by Lokesh
"""If you are running Pygame on a UNIX system, like a Linux server, try using a DummyVideoDriver"""
os.environ["SDL_VIDEODRIVER"] = "dummy"

# %%
# These define the set of attriutes to consider while generating the dataset
Colors = {'red': (220, 20, 60), 'orange': (255,165,0), 'Yellow': (255,255,0), 'green': (0,128,0), 'cyan' : (0,255,255),
         'blue': (0,0,255), 'purple': (128,0,128), 'pink': (255,192,203), 'chocolate': (210,105,30), 'silver': (192,192,192)}
Sizes = {'small': 10, 'medium': 20, 'large': 30}
All_fonts = pygame.font.get_fonts()
useless_fonts = ['notocoloremoji', 'droidsansfallback', 'gubbi', 'kalapi', 'lklug',  'mrykacstqurn', 'ori1uni','pothana2000','vemana2000',
                'navilu', 'opensymbol', 'padmmaa', 'raghumalayalam', 'saab', 'samyakdevanagari']
useless_fontsets = ['kacst', 'lohit', 'sam']

def throw_useless_fonts():
    for useless_font in useless_fonts:
        if useless_font in All_fonts:
            All_fonts.remove(useless_font)
    temp = All_fonts.copy()
    for useless_font in temp: # check every one
        for set in useless_fontsets:
            if set in useless_font:
                try:
                    All_fonts.remove(useless_font)
                except:
                    print(useless_font)
throw_useless_fonts()


# %%
# Hyperparameters for generating the dataset
cap_letters = list(range(65, 91))
small_letters = list(range(97, 123))
Letters = small_letters + cap_letters
Integers = list(range(48, 58))
target_chars = small_letters
words = []
img_size = 32

max_w_len = 1

'''
dfs takes a list L containing the single characters and use them to construct words of all permutations of characters 
with a maximum length of max_w_len
dfs also takes an empty string as a starting point
'''
def dfs(w, L):
    if len(w)==max_w_len:
        return
    for l in L:
        words.append(w+chr(l))
        dfs(w+chr(l), L)
dfs('', target_chars)

# directory to save the dataset
font_dir = './fonts'
if not os.path.exists(font_dir):
    os.makedirs(font_dir)

# initialize pygame and start to generate the dataset
pygame.init()
screen = pygame.display.set_mode((img_size, img_size)) # image size Fix(128 * 128)
cnt = 0

font_colors = ["red", "orange", "green", "cyan"]
font_backs = Colors.keys()

images = []
betas = []
labels = []

def create_idxdict(items):
    idx_dict = {}
    for idx, entry in enumerate(items):
        idx_dict[entry] = idx
    return idx_dict

candidate_fonts =  np.random.choice(All_fonts, size=7, replace=False)

words_dict, size_dict, fcolor_dict, bcolor_dict, font_dict = create_idxdict(words), create_idxdict(Sizes.keys()), \
    create_idxdict(font_colors), create_idxdict(font_backs), create_idxdict(candidate_fonts) 

for i, word in enumerate(words):
    print(f"Generating {i}th word: {word}")
    for size in Sizes.keys():  # 2nd round for size
        for font_color in font_colors: #Colors.keys():  # 3rd round for font_color
            for back_color in font_backs:
                for font in candidate_fonts:
                    '''This is code for generating font images'''
                    if not font_color == back_color:
                        try:
                            screen.fill(Colors[back_color]) # background color
                            selected_letter = word
                            selected_font = pygame.font.SysFont(font, Sizes[size]) # size and bold or not
                            font_size = selected_font.size(selected_letter);

                            rtext = selected_font.render(selected_letter, True, Colors[font_color], Colors[back_color])
                            drawX = img_size / 2 - (font_size[0] / 2.0)
                            drawY = img_size / 2 - (font_size[1] / 2.0)
                            screen.blit(rtext, (drawX, drawY)) # because

                            imgdata = pygame.surfarray.array3d(screen)
                            imgdata = imgdata.swapaxes(0,1)
                            
                            images.append(imgdata)
                            labels.append(words_dict[word])
                            beta = np.array([font_dict[font], fcolor_dict[font_color], size_dict[size], bcolor_dict[back_color]])
                            betas.append(beta)

                            assert len(images) == len(betas) and len(betas) == len(labels), "Why are the numbers inconsistent?"

                        except:
                            print(word, size, font_color, back_color, font)
                    else:
                        continue

print(len(images), len(betas), len(labels))

with open("fonts/fonts_dataset.pkl", "wb") as file:
    pkl.dump((images, betas, labels), file)                        

print('finished')