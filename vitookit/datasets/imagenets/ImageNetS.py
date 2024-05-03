import random
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import torch
import os
import numpy as np
import cv2
import numpy as np
from PIL import Image


def get_mask_of_class(mask, v):
    """
    Get binary mask of v-th class.
    :param mask (numpy array, uint8): semantic segmentation mask
    :param v (int): the index of given class
    :return: binary mask of v-th class
    """
    mask_v = (mask == v) * 255
    return mask_v.astype(np.uint8)


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

class ImageNetSValidation(VisionDataset):
    def __init__(self, root, subset,transform=None, target_transform=None,boundary_width=0.03,crop_size=(224,224)):
        """Validation set of ImageNet-S

        Args:
            root (str): the root directory of ImageNet-S
            subset (str): one of ['50', '300', '919']
            transform (Callable, optional): The transformation for images. Defaults to None.
            target_transform (Callable, optional): The transformation for masks. Defaults to None.
            boundary_width (float, optional): The width of boundary. Defaults to 0.03.
            crop_size (tuple, optional): The image size. Defaults to (224,224).
        """
        self.boundary_width=boundary_width
        self.crop_size = crop_size
        
        assert subset in ['50', '300', '919'], 'invalid subset: '+ subset
        split='validation'
        root = os.path.expanduser(root)
        image_root = os.path.join(root, 'ImageNetS{0}'.format(subset),split)
        label_root = os.path.join(root, 'ImageNetS{0}'.format(subset),'validation-segmentation')
         
        super(ImageNetSValidation, self).__init__(image_root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        
        self.image_lst = []
        self.label_lst = []
        self.params = {
            'num_classes': int(subset),
            'classes': f'classes_{subset}',
            'names': f'ImageNetS_im{subset}_{split}.txt'
        }

        with open(os.path.join(os.path.dirname(__file__),'names',self.params['names']), 'r') as f:
            names = f.read().splitlines()

            for name in names:
                img, label = name.split(' ')
                self.image_lst.append(os.path.join(image_root, img))
                self.label_lst.append(os.path.join(label_root, label))

        self.classes_50 = "goldfish, tiger shark, goldfinch, tree frog, kuvasz, red fox, siamese cat, american black bear, ladybug, sulphur butterfly, wood rabbit, hamster, wild boar, gibbon, african elephant, giant panda, airliner, ashcan, ballpoint, beach wagon, boathouse, bullet train, cellular telephone, chest, clog, container ship, digital watch, dining table, golf ball, grand piano, iron, lab coat, mixing bowl, motor scooter, padlock, park bench, purse, streetcar, table lamp, television, toilet seat, umbrella, vase, water bottle, water tower, yawl, street sign, lemon, carbonara, agaric"
        self.classes_300 = "tench, goldfish, tiger shark, hammerhead, electric ray, ostrich, goldfinch, house finch, indigo bunting, kite, common newt, axolotl, tree frog, tailed frog, mud turtle, banded gecko, american chameleon, whiptail, african chameleon, komodo dragon, american alligator, triceratops, thunder snake, ringneck snake, king snake, rock python, horned viper, harvestman, scorpion, garden spider, tick, african grey, lorikeet, red-breasted merganser, wallaby, koala, jellyfish, sea anemone, conch, fiddler crab, american lobster, spiny lobster, isopod, bittern, crane, limpkin, bustard, albatross, toy terrier, afghan hound, bluetick, borzoi, irish wolfhound, whippet, ibizan hound, staffordshire bullterrier, border terrier, yorkshire terrier, lakeland terrier, giant schnauzer, standard schnauzer, scotch terrier, lhasa, english setter, clumber, english springer, welsh springer spaniel, kuvasz, kelpie, doberman, miniature pinscher, malamute, pug, leonberg, great pyrenees, samoyed, brabancon griffon, cardigan, coyote, red fox, kit fox, grey fox, persian cat, siamese cat, cougar, lynx, tiger, american black bear, sloth bear, ladybug, leaf beetle, weevil, bee, cicada, leafhopper, damselfly, ringlet, cabbage butterfly, sulphur butterfly, sea cucumber, wood rabbit, hare, hamster, wild boar, hippopotamus, bighorn, ibex, badger, three-toed sloth, orangutan, gibbon, colobus, spider monkey, squirrel monkey, madagascar cat, indian elephant, african elephant, giant panda, barracouta, eel, coho, academic gown, accordion, airliner, ambulance, analog clock, ashcan, backpack, balloon, ballpoint, barbell, barn, bassoon, bath towel, beach wagon, bicycle-built-for-two, binoculars, boathouse, bonnet, bookcase, bow, brass, breastplate, bullet train, cannon, can opener, carpenter's kit, cassette, cellular telephone, chain saw, chest, china cabinet, clog, combination lock, container ship, corkscrew, crate, crock pot, digital watch, dining table, dishwasher, doormat, dutch oven, electric fan, electric locomotive, envelope, file, folding chair, football helmet, freight car, french horn, fur coat, garbage truck, goblet, golf ball, grand piano, half track, hamper, hard disc, harmonica, harvester, hook, horizontal bar, horse cart, iron, jack-o'-lantern, lab coat, ladle, letter opener, liner, mailbox, megalith, military uniform, milk can, mixing bowl, monastery, mortar, mosquito net, motor scooter, mountain bike, mountain tent, mousetrap, necklace, nipple, ocarina, padlock, palace, parallel bars, park bench, pedestal, pencil sharpener, pickelhaube, pillow, planetarium, plastic bag, polaroid camera, pole, pot, purse, quilt, radiator, radio, radio telescope, rain barrel, reflex camera, refrigerator, rifle, rocking chair, rubber eraser, rule, running shoe, sewing machine, shield, shoji, ski, ski mask, slot, soap dispenser, soccer ball, sock, soup bowl, space heater, spider web, spindle, sports car, steel arch bridge, stethoscope, streetcar, submarine, swimming trunks, syringe, table lamp, tank, teddy, television, throne, tile roof, toilet seat, trench coat, trimaran, typewriter keyboard, umbrella, vase, volleyball, wardrobe, warplane, washer, water bottle, water tower, whiskey jug, wig, wine bottle, wok, wreck, yawl, yurt, street sign, traffic light, consomme, ice cream, bagel, cheeseburger, hotdog, mashed potato, spaghetti squash, bell pepper, cardoon, granny smith, strawberry, lemon, carbonara, burrito, cup, coral reef, yellow lady's slipper, buckeye, agaric, gyromitra, earthstar, bolete"
        self.classes_919 = "house finch, stupa, agaric, hen-of-the-woods, wild boar, kit fox, desk, beaker, spindle, lipstick, cardoon, ringneck snake, daisy, sturgeon, scorpion, pelican, bustard, rock crab, rock beauty, minivan, menu, thunder snake, zebra, partridge, lacewing, starfish, italian greyhound, marmot, cardigan, plate, ballpoint, chesapeake bay retriever, pirate, potpie, keeshond, dhole, waffle iron, cab, american egret, colobus, radio telescope, gordon setter, mousetrap, overskirt, hamster, wine bottle, bluetick, macaque, bullfrog, junco, tusker, scuba diver, pool table, samoyed, mailbox, purse, monastery, bathtub, window screen, african crocodile, traffic light, tow truck, radio, recreational vehicle, grey whale, crayfish, rottweiler, racer, whistle, pencil box, barometer, cabbage butterfly, sloth bear, rhinoceros beetle, guillotine, rocking chair, sports car, bouvier des flandres, border collie, fiddler crab, slot, go-kart, cocker spaniel, plate rack, common newt, tile roof, marimba, moped, terrapin, oxcart, lionfish, bassinet, rain barrel, american black bear, goose, half track, kite, microphone, shield, mexican hairless, measuring cup, bubble, platypus, saint bernard, police van, vase, lhasa, wardrobe, teapot, hummingbird, revolver, jinrikisha, mailbag, red-breasted merganser, assault rifle, loudspeaker, fig, american lobster, can opener, arctic fox, broccoli, long-horned beetle, television, airship, black stork, marmoset, panpipe, drumstick, knee pad, lotion, french loaf, throne, jeep, jersey, tiger cat, cliff, sealyham terrier, strawberry, minibus, goldfinch, goblet, burrito, harp, tractor, cornet, leopard, fly, fireboat, bolete, barber chair, consomme, tripod, breastplate, pineapple, wok, totem pole, alligator lizard, common iguana, digital clock, bighorn, siamese cat, bobsled, irish setter, zucchini, crock pot, loggerhead, irish wolfhound, nipple, rubber eraser, impala, barbell, snow leopard, siberian husky, necklace, manhole cover, electric fan, hippopotamus, entlebucher, prison, doberman, ruffed grouse, coyote, toaster, puffer, black swan, schipperke, file, prairie chicken, hourglass, greater swiss mountain dog, pajama, ear, pedestal, viaduct, shoji, snowplow, puck, gyromitra, birdhouse, flatworm, pier, coral reef, pot, mortar, polaroid camera, passenger car, barracouta, banded gecko, black-and-tan coonhound, safe, ski, torch, green lizard, volleyball, brambling, solar dish, lawn mower, swing, hyena, staffordshire bullterrier, screw, toilet tissue, velvet, scale, stopwatch, sock, koala, garbage truck, spider monkey, afghan hound, chain, upright, flagpole, tree frog, cuirass, chest, groenendael, christmas stocking, lakeland terrier, perfume, neck brace, lab coat, carbonara, porcupine, shower curtain, slug, pitcher, flat-coated retriever, pekinese, oscilloscope, church, lynx, cowboy hat, table lamp, pug, crate, water buffalo, labrador retriever, weimaraner, giant schnauzer, stove, sea urchin, banjo, tiger, miniskirt, eft, european gallinule, vending machine, miniature schnauzer, maypole, bull mastiff, hoopskirt, coffeepot, four-poster, safety pin, monarch, beer glass, grasshopper, head cabbage, parking meter, bonnet, chiffonier, great dane, spider web, electric locomotive, scotch terrier, australian terrier, honeycomb, leafhopper, beer bottle, mud turtle, lifeboat, cassette, potter's wheel, oystercatcher, space heater, coral fungus, sunglass, quail, triumphal arch, collie, walker hound, bucket, bee, komodo dragon, dugong, gibbon, trailer truck, king crab, cheetah, rifle, stingray, bison, ipod, modem, box turtle, motor scooter, container ship, vestment, dingo, radiator, giant panda, nail, sea slug, indigo bunting, trimaran, jacamar, chimpanzee, comic book, odometer, dishwasher, bolo tie, barn, paddlewheel, appenzeller, great white shark, green snake, jackfruit, llama, whippet, hay, leaf beetle, sombrero, ram, washbasin, cup, wall clock, acorn squash, spotted salamander, boston bull, border terrier, doormat, cicada, kimono, hand blower, ox, meerkat, space shuttle, african hunting dog, violin, artichoke, toucan, bulbul, coucal, red wolf, seat belt, bicycle-built-for-two, bow tie, pretzel, bedlington terrier, albatross, punching bag, cocktail shaker, diamondback, corn, ant, mountain bike, walking stick, standard schnauzer, power drill, cardigan, accordion, wire-haired fox terrier, streetcar, beach wagon, ibizan hound, hair spray, car mirror, mountain tent, trench coat, studio couch, pomeranian, dough, corkscrew, broom, parachute, band aid, water tower, teddy, fire engine, hornbill, hotdog, theater curtain, crane, malinois, lion, african elephant, handkerchief, caldron, shopping basket, gown, wolf spider, vizsla, electric ray, freight car, pembroke, feather boa, wallet, agama, hard disc, stretcher, sorrel, trilobite, basset, vulture, tarantula, hermit crab, king snake, robin, bernese mountain dog, ski mask, fountain pen, combination lock, yurt, clumber, park bench, baboon, kuvasz, centipede, tabby, steam locomotive, badger, irish water spaniel, picket fence, gong, canoe, swimming trunks, submarine, echidna, bib, refrigerator, hammer, lemon, admiral, chihuahua, basenji, pinwheel, golfcart, bullet train, crib, muzzle, eggnog, old english sheepdog, tray, tiger beetle, electric guitar, peacock, soup bowl, wallaby, abacus, dalmatian, harvester, aircraft carrier, snowmobile, welsh springer spaniel, affenpinscher, oboe, cassette player, pencil sharpener, japanese spaniel, plunger, black widow, norfolk terrier, reflex camera, ice bear, redbone, mongoose, warthog, arabian camel, bittern, mixing bowl, tailed frog, scabbard, castle, curly-coated retriever, garden spider, folding chair, mouse, prayer rug, red fox, toy terrier, leonberg, lycaenid, poncho, goldfish, red-backed sandpiper, holster, hair slide, coho, komondor, macaw, maltese dog, megalith, sarong, green mamba, sea lion, water ouzel, bulletproof vest, sulphur-crested cockatoo, scottish deerhound, steel arch bridge, catamaran, brittany spaniel, redshank, otter, brabancon griffon, balloon, rule, planetarium, trombone, mitten, abaya, crash helmet, milk can, hartebeest, windsor tie, irish terrier, african chameleon, matchstick, water bottle, cloak, ground beetle, ashcan, crane, gila monster, unicycle, gazelle, wombat, brain coral, projector, custard apple, proboscis monkey, tibetan mastiff, mosque, plastic bag, backpack, drum, norwich terrier, pizza, carton, plane, gorilla, jigsaw puzzle, forklift, isopod, otterhound, vacuum, european fire salamander, apron, langur, boxer, african grey, ice lolly, toilet seat, golf ball, titi, drake, ostrich, magnetic compass, great pyrenees, rhodesian ridgeback, buckeye, dungeness crab, toy poodle, ptarmigan, amphibian, monitor, school bus, schooner, spatula, weevil, speedboat, sundial, borzoi, bassoon, bath towel, pill bottle, acorn, tick, briard, thimble, brass, white wolf, boathouse, yawl, miniature pinscher, barn spider, jean, water snake, dishrag, yorkshire terrier, hammerhead, typewriter keyboard, papillon, ocarina, washer, standard poodle, china cabinet, steel drum, swab, mobile home, german short-haired pointer, saluki, bee eater, rock python, vine snake, kelpie, harmonica, military uniform, reel, thatch, maraca, tricycle, sidewinder, parallel bars, banana, flute, paintbrush, sleeping bag, yellow lady's slipper, three-toed sloth, white stork, notebook, weasel, tiger shark, football helmet, madagascar cat, dowitcher, wreck, king penguin, lighter, timber wolf, racket, digital watch, liner, hen, suspension bridge, pillow, carpenter's kit, butternut squash, sandal, sussex spaniel, hip, american staffordshire terrier, flamingo, analog clock, black and gold garden spider, sea cucumber, indian elephant, syringe, lens cap, missile, cougar, diaper, chambered nautilus, garter snake, anemone fish, organ, limousine, horse cart, jaguar, frilled lizard, crutch, sea anemone, guenon, meat loaf, slide rule, saltshaker, pomegranate, acoustic guitar, shopping cart, drilling platform, nematode, chickadee, academic gown, candle, norwegian elkhound, armadillo, horizontal bar, orangutan, obelisk, stone wall, cannon, rugby ball, ping-pong ball, window shade, trolleybus, ice cream, pop bottle, cock, harvestman, leatherback turtle, killer whale, spaghetti squash, chain saw, stinkhorn, espresso maker, loafer, bagel, ballplayer, skunk, chainlink fence, earthstar, whiptail, barrel, kerry blue terrier, triceratops, chow, grey fox, sax, binoculars, ladybug, silky terrier, gas pump, cradle, whiskey jug, french bulldog, eskimo dog, hog, hognose snake, pickup, indian cobra, hand-held computer, printer, pole, bald eagle, american alligator, dumbbell, umbrella, mink, shower cap, tank, quill, fox squirrel, ambulance, lesser panda, frying pan, letter opener, hook, strainer, pick, dragonfly, gar, piggy bank, envelope, stole, ibex, american chameleon, bearskin, microwave, petri dish, wood rabbit, beacon, dung beetle, warplane, ruddy turnstone, knot, fur coat, hamper, beagle, ringlet, mask, persian cat, cellular telephone, american coot, apiary, shovel, coffee mug, sewing machine, spoonbill, padlock, bell pepper, great grey owl, squirrel monkey, sulphur butterfly, scoreboard, bow, malamute, siamang, snail, remote control, sea snake, loupe, model t, english setter, dining table, face powder, tench, jack-o'-lantern, croquet ball, water jug, airedale, airliner, guinea pig, hare, damselfly, thresher, limpkin, buckle, english springer, boa constrictor, french horn, black-footed ferret, shetland sheepdog, capuchin, cheeseburger, miniature poodle, spotlight, wooden spoon, west highland white terrier, wig, running shoe, cowboy boot, brown bear, iron, brassiere, magpie, gondola, grand piano, granny smith, mashed potato, german shepherd, stethoscope, cauliflower, soccer ball, pay-phone, jellyfish, cairn, polecat, trifle, photocopier, shih-tzu, orange, guacamole, hatchet, cello, egyptian cat, basketball, moving van, mortarboard, dial telephone, street sign, oil filter, beaver, spiny lobster, chime, bookcase, chiton, black grouse, jay, axolotl, oxygen mask, cricket, worm fence, indri, cockroach, mushroom, dandie dinmont, tennis ball, howler monkey, rapeseed, tibetan terrier, newfoundland, dutch oven, paddle, joystick, golden retriever, blenheim spaniel, mantis, soft-coated wheaten terrier, little blue heron, convertible, bloodhound, palace, medicine chest, english foxhound, cleaver, sweatshirt, mosquito net, soap dispenser, ladle, screwdriver, fire screen, binder, suit, barrow, clog, cucumber, baseball, lorikeet, conch, quilt, eel, horned viper, night snake, angora, pickelhaube, gasmask, patas"
        self.classes_50 = ['background'] + self.classes_50.split(', ')
        self.classes_300 = ['background'] + self.classes_300.split(', ')
        self.classes_919 = ['background'] + self.classes_919.split(', ')
        if subset == '50':
            self.classes = self.classes_50
        elif subset == '300':
            self.classes = self.classes_300
        else:
            self.classes = self.classes_919

    def __getitem__(self, item):
        x = self.loader(self.image_lst[item])
        gt = self.loader(self.label_lst[item])
        x,gt = self.resize_crop(x,gt,self.crop_size)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            gt = self.target_transform(gt)
        else:
            gt = np.array(gt)
            gt = gt[:, :, 1] * 256 + gt[:, :, 0]
            

        # Get boundary mask for each class.
        boundary_gt = self.get_boundary_mask(gt + 1)

        gt = torch.from_numpy(gt.astype(np.float))
        boundary_gt = torch.from_numpy(boundary_gt.astype(np.float))

        return x, gt , boundary_gt
    
        
    def resize_crop(self, x:Image, gt, crop_size):
        w, h = x.size
        th, tw = crop_size
        if w < tw or h < th:
            scale = max((tw) / w, (1*th) / h)
            x = x.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            gt = gt.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            w, h = x.size
        
        # scale down
        if w > tw and h > th:
            scale = min((tw) / w, (th) / h)
            x = x.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            gt = gt.resize((int(w*scale), int(h*scale)), Image.NEAREST)
            w, h = x.size
        
        j = (w - tw)//2
        i =( h - th)//2
            
        x = x.crop((j, i, j+tw, i+th))
        gt = gt.crop((j, i, j+tw, i+th))
        return x, gt
    
    def get_boundary_mask(self, mask):
        boundary = np.zeros_like(mask).astype(mask.dtype)
        for v in np.unique(mask):
            mask_v =get_mask_of_class(mask, v)
            boundary_v = mask_to_boundary(mask_v, dilation_ratio=self.boundary_width)
            boundary += (boundary_v > 0) * v
        return boundary

    def __len__(self):
        return len(self.label_lst)