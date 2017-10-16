## http://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html

import os
import sys
from PIL import Image, ImageDraw
import numpy as np
import numpy.random as nprandom
import random
import math
from scipy import misc, ndimage
import shutil
if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle
import gzip

import pdb

def read_dataset(filename='alldata.data'):
    if filename[-3:] == '.gz':
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')
    if sys.version_info[0] < 3:
        dataxy = cPickle.load(f)
    else:
        dataxy = pickle.load(f, encoding='latin1')
    f.close()
    return dataxy

def save_dataset(filename='alldata.data.gz', data=None):
    if filename[-3:] == '.gz':
        f = gzip.open(filename, 'wb')
    else:
        f = open(filename, 'wb')
    if sys.version_info[0] < 3:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        pickle.dump(data, f, protocol=2)
    f.close()

class Shape(object):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Shape'
        self.size = size
        self.center = center
        
    def overlap(self, oshape):
        
        left = self.center[0]-self.size/2.
        right = self.center[0]+self.size/2.
        up = self.center[1]-self.size/2.
        down = self.center[1]+self.size/2.
        
        oleft = oshape.center[0]-oshape.size/2.
        oright = oshape.center[0]+oshape.size/2.
        oup = oshape.center[1]-oshape.size/2.
        odown = oshape.center[1]+oshape.size/2.
        # pdb.set_trace()
        if left > oright or right < oleft or up > odown or down < oup:
            return False
        else:
            return True
        
    @staticmethod
    def get_name():
        return self.name
        
class Triangle(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Triangle'
        self.size = size
        self.center = center
        
    
class Rectangle(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Rectangle'
        self.size = size
        self.center = center
        
class Ellipse(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Ellipse'
        self.size = size
        self.center = center
        
class Flower4(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Flower4'
        self.size = size
        self.center = center
        
class Flower2(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Flower2'
        self.size = size
        self.center = center
        
class Hexagram(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Hexagram'
        self.size = size
        self.center = center
        
        
class Symmetry(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Symmetry'
        self.size = size
        self.center = center

class Asymmetry(Shape):
    def __init__(self, center=[0,0], size=0):
        self.name = 'Asymmetry'
        self.size = size
        self.center = center

        
def drawTriangle(im, components, center, size, fill, equilateral=True, rotate=True, rescale=False):
    obj = Triangle(center, size)
    for other in components:
        if obj.overlap(other):
            return None
    draw = ImageDraw.Draw(im)
    vertices = []
    radius = size/2.
    if equilateral:
        ## scale
        scale = 1.0
        if rescale:
            scale = nprandom.uniform(0.5, 1.0)
        radius = radius*scale
        
        vertices.append((center[0], center[1]+radius))
        vertices.append((center[0]+radius * math.cos(7*math.pi /6.), center[1]+radius * math.sin(7*math.pi /6.)))
        vertices.append((center[0]+radius * math.cos(-math.pi /6.), center[1]+radius * math.sin(-math.pi /6.)))
        ## rotate
        if rotate:
            angle = nprandom.uniform(0, 2*math.pi /3.)
            R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            newV = np.dot(np.asarray(vertices)-center, R.T) + center
            vertices = newV.flatten().tolist()
        
        obj.size = size*scale # update obj size
    else:
        angle = 0
        angRange = math.pi /3.
        
        for i in range(3):                         # Do this 3 times
            angle = nprandom.uniform(angle+angRange, angle+2*angRange)
            x = round(center[0] + radius * math.cos(angle))
            y = round(center[1] + radius * math.sin(angle))
            vertices.append((x, y))
    draw.polygon(vertices, fill = fill)
    return obj

def drawRectangule(im, components, center, size, fill, rotate=True, rescale=False):
    obj = Rectangle(center, size)
    for other in components:
        if obj.overlap(other):
            return None
    draw = ImageDraw.Draw(im)
    vertices = []
    
    if rescale:
        w = nprandom.randint(size/2, size)
        h = nprandom.randint(size/2, size)
    else:
        w = size
        h = size
    x0 = center[0]-w/2
    x1 = x0 + w
    y0 = center[1]-h/2
    y1 = y0 + h
    
    vertices = [(x0, y0), (x0,y1), (x1,y1), (x1,y0)]
    center = np.array([(x0+x1)/2.0, (y0+y1)/2.0])
    ## rotate
    if rotate:
        # angle = nprandom.uniform(-math.pi/4., math.pi/4.)
        angle = nprandom.uniform(-math.pi/10., math.pi/10.)
        R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        newV = np.round(np.dot(np.asarray(vertices)-center, R.T) + center).astype(int)
        vertices = newV.flatten().tolist()
        
    draw.polygon(vertices, fill=fill)
    
    obj.size = np.ceil(np.sqrt(w**2+h**2)) # update obj size
    
    return obj
    
def drawEllipse(im, components, center, size, fill, rescale=False):
    obj = Ellipse(center, size)
    for other in components:
        if obj.overlap(other):
            return None
            
    draw = ImageDraw.Draw(im)
    
    w = nprandom.randint(size/2, size) if rescale else size
    h = w
    # h = nprandom.randint(size/2, size)
    x0 = center[0]-w/2
    x1 = x0 + w
    y0 = center[1]-h/2
    y1 = y0 + h
    
    draw.ellipse([x0,y0,x1,y1], fill = fill)
    
    obj.size = max(w,h) # update obj size
    
    return obj

def drawFlower4(im, components, center, size, fill, rotate=False, rescale=False):
    obj = Flower4(center, size)
    for other in components:
        if obj.overlap(other):
            return None
            
    draw = ImageDraw.Draw(im)
    
    w = nprandom.randint(size/2, size) if rescale else size
    h = w
    # h = nprandom.randint(size/2, size)
    x0 = center[0]-w/2
    x1 = x0 + w
    y0 = center[1]-h/2
    y1 = y0 + h
    
    ts = [t/100.0 for t in range(101)]
    
    points = [(center[0],center[1]), (center[0]+w/3,center[1]-h/2), (center[0]-w/3,center[1]-h/2), (center[0],center[1]), ]
    bezier = make_bezier(points)
    bpts = bezier(ts)
    # draw.polygon(bpts, fill = fill)
    
    points = [(center[0],center[1]), (center[0]+w/3,center[1]+h/2), (center[0]-w/3,center[1]+h/2), (center[0],center[1]), ]
    bezier = make_bezier(points)
    bpts.extend(bezier(ts))
    # draw.polygon(bpts, fill = fill)
    
    points = [(center[0],center[1]), (center[0]-w/2,center[1]-h/3), (center[0]-w/2,center[1]+h/3), (center[0],center[1]), ]
    bezier = make_bezier(points)
    bpts.extend(bezier(ts))
    # draw.polygon(bpts, fill = fill)
    
    points = [(center[0],center[1]), (center[0]+w/2,center[1]-h/3), (center[0]+w/2,center[1]+h/3), (center[0],center[1]), ]
    bezier = make_bezier(points)
    bpts.extend(bezier(ts))
    
    if rotate:
        angle = nprandom.uniform(-math.pi/4., math.pi/4.)
        R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        newV = np.dot(np.asarray(bpts)-center, R.T) + center
        bpts = newV.flatten().tolist()
        
    
    draw.polygon(bpts, fill = fill)
    
    ## draw a center ball
    # x0 = center[0]-w/6
    # x1 = x0 + w/3
    # y0 = center[1]-h/6
    # y1 = y0 + h/3
    
    # draw.ellipse([x0,y0,x1,y1], fill = fill)
    
    obj.size = max(w,h) # update obj size
    
    return obj
    
def drawFlower2(im, components, center, size, fill, rotate=False, rescale=False):
    obj = Flower2(center, size)
    for other in components:
        if obj.overlap(other):
            return None
            
    draw = ImageDraw.Draw(im)
    
    w = nprandom.randint(size/2, size) if rescale else size
    h = w
    # h = nprandom.randint(size/2, size)
    x0 = center[0]-w/2
    x1 = x0 + w
    y0 = center[1]-h/2
    y1 = y0 + h
    
    ts = [t/100.0 for t in range(101)]
    
    if nprandom.rand() < 0.5:
        points = [(center[0],center[1]), (center[0]+w/3,center[1]-h/2), (center[0]-w/3,center[1]-h/2), (center[0],center[1]), ]
        bezier = make_bezier(points)
        vertices = bezier(ts)
        
        points = [(center[0],center[1]), (center[0]+w/3,center[1]+h/2), (center[0]-w/3,center[1]+h/2), (center[0],center[1]), ]
        bezier = make_bezier(points)
        vertices.extend(bezier(ts))
    else:
        points = [(center[0],center[1]), (center[0]-w/2,center[1]-h/3), (center[0]-w/2,center[1]+h/3), (center[0],center[1]), ]
        bezier = make_bezier(points)
        vertices = bezier(ts)
        
        points = [(center[0],center[1]), (center[0]+w/2,center[1]-h/3), (center[0]+w/2,center[1]+h/3), (center[0],center[1]), ]
        bezier = make_bezier(points)
        vertices.extend(bezier(ts))
    
    if rotate:
        angle = nprandom.uniform(-math.pi/4., math.pi/4.)
        R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        newV = np.dot(np.asarray(vertices)-center, R.T) + center
        vertices = newV.flatten().tolist()
        
    draw.polygon(vertices, fill = fill)
    
    obj.size = max(w,h) # update obj size
    
    return obj
    
def drawHexagram(im, components, center, size, fill, rotate=False, rescale=False):
    obj = Hexagram(center, size)
    for other in components:
        if obj.overlap(other):
            return None
    draw = ImageDraw.Draw(im)
    vertices = []
    radius = size/2.
    
    ## scale
    scale = 1.0
    if rescale:
        scale = nprandom.uniform(0.8, 1.0)
    radius = radius*scale
    
    vertices.append((center[0], center[1]+radius))
    vertices.append((center[0]+radius * math.cos(7*math.pi /6.), center[1]+radius * math.sin(7*math.pi /6.)))
    vertices.append((center[0]+radius * math.cos(-math.pi /6.), center[1]+radius * math.sin(-math.pi /6.)))
    
    ## rotate
    if rotate:
        angle = nprandom.uniform(0, 2*math.pi /3.)
        R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        newV = np.dot(np.asarray(vertices)-center, R.T) + center
        vertices = newV.flatten().tolist()
    
    draw.polygon(vertices, fill = fill)
    
    vertices = [(center[0], center[1]-radius)]
    vertices.append((center[0]+radius * math.cos(7*math.pi /6.), center[1]-radius * math.sin(7*math.pi /6.)))
    vertices.append((center[0]+radius * math.cos(-math.pi /6.), center[1]-radius * math.sin(-math.pi /6.)))
    
    ## rotate
    if rotate:
        newV = np.dot(np.asarray(vertices)-center, R.T) + center
        vertices = newV.flatten().tolist()
    
    draw.polygon(vertices, fill = fill)
    
    obj.size = size*scale # update obj size
    return obj
    
def drawEraser(im, center, size, fill=0):
    obj = Ellipse(center, size)
    
    draw = ImageDraw.Draw(im)
    
    w = size
    h = w
    # h = nprandom.randint(size/2, size)
    x0 = center[0]-w/2
    x1 = x0 + w
    y0 = center[1]-h/2
    y1 = y0 + h
    
    draw.ellipse([x0,y0,x1,y1], fill = fill)
    
    return obj
    
def drawEraser2(im, center, size, fill=0):
    
    v = im.getpixel(tuple(center))
    W,H = im.size
    
    w = size
    h = w
    x0 = int(center[0]-w/2)
    x1 = int(x0 + w +1)
    y0 = int(center[1]-h/2)
    y1 = int(y0 + h +1)
    x0 = max(0, x0)
    x1 = min(W, x1)
    y0 = max(0, y0)
    y1 = min(H, y1)
    for x in range(x0,x1):
        for y in range(y0,y1):
            if v == im.getpixel((x,y)):
                im.putpixel((x,y), 0)
                
    obj = Shape(center, size) # dummy shape
    return obj
    
# img1 = Image.new('RGBA', (255, 255)) # Use RGBA
# img2 = Image.new('RGBA', (255, 255)) # Use RGBA
# draw1 = ImageDraw.Draw(img1)
# draw2 = ImageDraw.Draw(img2)

# draw1.polygon([(0, 0), (0, 255), (255, 255), (255, 0)], fill = (255,255,255,255))

# transparence = 100 # Define transparency for the triangle.
# draw2.polygon([(1,1), (20, 100), (100,20)], fill = (200, 0, 0, transparence))

# img = Image.alpha_composite(img1, img2)
# img.save("my_pic.png", 'PNG')

def testDraw():
    import matplotlib.pyplot as plt
    
    im = Image.new('L', (200, 200))

    componentList = []

    obj = None
    while obj is None:
        obj = drawFlower4(im, componentList, nprandom.randint(30,170,size=(2,)), nprandom.randint(30,40), nprandom.randint(128,255), rotate=False)
        if obj is not None:
            componentList.append(obj)
    
    obj = None
    while obj is None:
        obj = drawFlower2(im, componentList, nprandom.randint(30,170,size=(2,)), nprandom.randint(30,40), nprandom.randint(128,255), rotate=False)
        if obj is not None:
            componentList.append(obj)
            
    obj = None
    while obj is None:
        obj = drawHexagram(im, componentList, nprandom.randint(30,170,size=(2,)), nprandom.randint(30,40), nprandom.randint(128,255), rotate=False)
        if obj is not None:
            componentList.append(obj)
            
    obj = None
    while obj is None:
        obj = drawTriangle(im, componentList, nprandom.randint(20,180,size=(2,)), nprandom.randint(20,30), nprandom.randint(128,255), equilateral=True, rotate=False)
        if obj is not None:
            componentList.append(obj)

    # pdb.set_trace()
    obj = None
    while obj is None:
        obj = drawRectangule(im, componentList, nprandom.randint(20,180,size=(2,)), nprandom.randint(20,30), nprandom.randint(128,255),rotate=False)
        if obj is not None:
            componentList.append(obj)
            
    obj = None
    while obj is None:
        obj = drawEllipse(im, componentList, nprandom.randint(20,180,size=(2,)), nprandom.randint(20,30), nprandom.randint(128,255))
        if obj is not None:
            componentList.append(obj)

    print(len(componentList))

    plt.imshow(im, cmap='gray')
    plt.show()

    # pdb.set_trace()

    for i in range(len(componentList)):
        print(componentList[i].center, componentList[i].size)

    # pdb.set_trace()

    ii = 0
    jj = 1
    componentList[ii].overlap(componentList[jj])

    #write to file
    im.save('test.png', "PNG")

def drawOneImage(result, size, nTri, nRect, nBall=0, nHex=0, nF4=0, nF2=0, nSym=0, nAsym=0, rotate=True, baseSize=20, sizerange=10, newAsym=False):
    im = Image.new('L', size)
    W, H = size
    componentList = []

    pad = (baseSize+20)/2
    
    for _ in range(nTri):
        obj = None
        while obj is None:
            obj = drawTriangle(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255), equilateral=True, rotate=rotate)
            if obj is not None:
                componentList.append(obj)
        
    for _ in range(nRect):
        obj = None
        while obj is None:
            obj = drawRectangule(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255), rotate=rotate)
            if obj is not None:
                componentList.append(obj)
            
    for _ in range(nBall):
        obj = None
        while obj is None:
            obj = drawEllipse(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255))
            if obj is not None:
                componentList.append(obj)

    for _ in range(nHex):
        obj = None
        while obj is None:
            obj = drawHexagram(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize+6,baseSize+6+sizerange), nprandom.randint(128,255), rotate=rotate)
            if obj is not None:
                componentList.append(obj)
                
    for _ in range(nF4):
        obj = None
        while obj is None:
            obj = drawFlower4(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize+6,baseSize+6+sizerange), nprandom.randint(128,255), rotate=rotate)
            if obj is not None:
                componentList.append(obj)
                
    for _ in range(nF2):
        obj = None
        while obj is None:
            obj = drawFlower2(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255), rotate=rotate)
            if obj is not None:
                componentList.append(obj)
                
    for _ in range(nSym):
        obj = None
        while obj is None:
            obj = drawSymmetricPattern(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255), rotate=rotate)
            if obj is not None:
                componentList.append(obj)
                
    for _ in range(nAsym):
        obj = None
        while obj is None:
            ## when training ds5
            if not newAsym:
                obj = drawAsymmetricPattern(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255), rotate=rotate)
            else: ## use this when testing ds5 new shape
                obj = drawAsymmetricPattern2(im, componentList, [nprandom.randint(pad,W-pad), nprandom.randint(pad,H-pad)], nprandom.randint(baseSize,baseSize+sizerange), nprandom.randint(128,255), rotate=rotate)
            if obj is not None:
                componentList.append(obj)
                
    # print(len(componentList))

    # for i in range(len(componentList)):
        # print componentList[i].center, componentList[i].size

    #write to file
    im.save(result, "PNG")
    
    return componentList
    
def packImages(rpath, result):
    dirs = os.listdir(rpath)
    data_x = []
    data_y = []
    for folder in dirs:
        subf = os.path.join(rpath, folder)
        if os.path.isdir(subf):
            try:
                label = int(folder)
            except:
                print('this is not a class subfolder, subfolder name must be integer')
                continue
            
            images = os.listdir(subf)
            images.sort(key = lambda x: int(os.path.basename(x)[:os.path.basename(x).rindex('.')])) # sort with filename (id)
            for image in images:
                im = misc.imread(os.path.join(subf, image))
                data_x.append(im.tolist())
                data_y.append(label)
    save_dataset(result, (data_x, data_y))
    
def unpackImages(datafile, rpath, listFile=[]):
    (data_x, data_y) = read_dataset(datafile)
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    path0 = os.path.join(rpath, '0')
    os.mkdir(path0)
    path1 = os.path.join(rpath, '1')
    os.mkdir(path1)
    
    flist = []
    if listFile != [] and os.path.isfile(listFile):
        with open(listFile, 'r') as thefile:
            for line in thefile:
                flist.append(line.strip('\n'))
        assert len(flist)*2 == len(data_y)
    
    i = 0
    for data,label in zip(data_x,data_y):
        subf = os.path.join(rpath, str(label))

        if flist != []:
            imagename = flist[i]
        else:
            imagename = str(i)+'.png'
        i+=1
        misc.imsave(os.path.join(subf, imagename), data)
    
'''
=============================================================================================
counting types
'''
def generate_count_type_data():
    generate_dataset_1(r'data\count_type')
    generate_challenge_1_newShape(r'data\count_type') # deliberate test 1
    generate_challenge_1_newSize(r'data\count_type') # deliberate test 2
    generate_dataset_1_newShape(r'data\count_type\additional_1')
    generate_dataset_1_newSize(r'data\count_type\additional_2')
    generate_challenge_1_newSize2(r'data\count_type\additional_2')
    
## generate data set 1, with 2 classes, simple shapes (triangle or rectangle) with different scales, rotates
## class 0: each image contains one type of shape
## class 1: each image contains two types of shape
def generate_dataset_1(rpath, size=[200,200], nsample=2000, rotate=True):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            choice = nprandom.rand()
            if choice < 1.0/2.0:
                nTri = nprandom.randint(2,7)
            else:
                nRect = nprandom.randint(2,7)
            
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # class 1
        for isample in range(nsample):
            nTri = nprandom.randint(1,4)
            nRect = nprandom.randint(1,4)
            nBall = 0
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    
## deliberate test 1
def generate_challenge_1_newShape(rpath, size=[200,200], nsample=2000, rotate=True, subset = 'break_shape'):
    
    componentList1 = []
    componentList2 = []
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    # class 0
    for isample in range(nsample):
        nHex = 0
        nF4 = 0
        nBall = 0
        choice = nprandom.rand()
        if choice < 1.0/3.0:
            nHex = nprandom.randint(2,7)
        elif choice < 2.0/3.0:
            nF4 = nprandom.randint(2,7)
        else:
            nBall = nprandom.randint(2,7)
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, 0, 0, nBall=nBall, nHex=nHex, nF4=nF4, rotate=rotate) )
        
    # class 1
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nHex = 0
        nF4 = 0
        nBall = 0
        choice = nprandom.rand()
        if choice < 1.0/10.:
            nTri = nprandom.randint(1,4)
            nRect = nprandom.randint(1,4)
        elif choice < 2.0/10.:
            nHex = nprandom.randint(1,4)
            nF4 = nprandom.randint(1,4)
        elif choice < 3.0/10.0:
            nBall = nprandom.randint(1,4)
            nHex = nprandom.randint(1,4)
        elif choice < 4.0/10.0:
            nTri = nprandom.randint(1,4)
            nHex = nprandom.randint(1,4)
        elif choice < 5.0/10.0:
            nRect = nprandom.randint(1,4)
            nHex = nprandom.randint(1,4)
        elif choice < 6.0/10.0:
            nF4 = nprandom.randint(1,4)
            nRect = nprandom.randint(1,4)
        elif choice < 7.0/10.0:
            nF4 = nprandom.randint(1,4)
            nTri = nprandom.randint(1,4)
        elif choice < 8.0/10.0:
            nTri = nprandom.randint(1,4)
            nBall = nprandom.randint(1,4)
        elif choice < 9.0/10.0:
            nRect = nprandom.randint(1,4)
            nBall = nprandom.randint(1,4)
        else:
            nF4 = nprandom.randint(1,4)
            nBall = nprandom.randint(1,4)
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nHex=nHex, nF4=nF4, rotate=rotate) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    
## deliberate test 2
def generate_challenge_1_newSize(rpath, size=[200,200], nsample=2000, rotate=True, subset = 'break_size_30_40'):
    
    componentList1 = []
    componentList2 = []
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    # class 0
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nHex = 0
        nF4 = 0
        nBall = 0
        choice = nprandom.rand()
        if choice < 0.5:
            nTri = nprandom.randint(2,5)
        else:
            nRect = nprandom.randint(2,5)
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nHex=nHex, nF4=nF4, rotate=rotate, baseSize=30) )
        
    # class 1
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nHex = 0
        nF4 = 0
        nBall = 0
        
        nTri = nprandom.randint(1,3)
        nRect = nprandom.randint(1,3)
        
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nHex=nHex, nF4=nF4, rotate=rotate, baseSize=30) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    
## additional test 1
def generate_dataset_1_newShape(rpath, size=[200,200], nsample=3000, rotate=True):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nF4 = 0
            nBall = 0
            choice = nprandom.rand()
            if choice < 1.0/3.0:
                nTri = nprandom.randint(2,7)
            elif choice < 2.0/3.0:
                nRect = nprandom.randint(2,7)
            else:
                # nF4 = nprandom.randint(2,7) # 0728: add F4 into training
                nBall = nprandom.randint(2,7)
            
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nF4=nF4, rotate=rotate) )
            
        # class 1
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nF4 = 0
            nBall = 0
            choice = nprandom.rand()
            if choice < 1.0/3.:
                nTri = nprandom.randint(1,4)
                nRect = nprandom.randint(1,4)
            elif choice < 2.0/3.:
                nTri = nprandom.randint(1,4)
                # nF4 = nprandom.randint(1,4)
                nBall = nprandom.randint(1,4)
            else:
                nRect = nprandom.randint(1,4)
                # nF4 = nprandom.randint(1,4)
                nBall = nprandom.randint(1,4)
            
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nF4=nF4, rotate=rotate) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## additional test 2
def generate_dataset_1_newSize(rpath, size=[200,200], nsample=3000, rotate=True):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            if nprandom.rand() < 0.5:
                baseSize = 20
                maxCnt = 7
            else:
                baseSize = 40
                maxCnt = 5
                
            choice = nprandom.rand()
            if choice < 1.0/2.0:
                nTri = nprandom.randint(2,maxCnt)
            else:
                nRect = nprandom.randint(2,maxCnt)
            
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, rotate=rotate, baseSize=baseSize, sizerange=5) )
            
        # class 1
        for isample in range(nsample):
            if nprandom.rand() < 0.5:
                baseSize = 20
                maxCnt = 4
            else:
                baseSize = 40
                maxCnt = 3
            nTri = nprandom.randint(1,maxCnt)
            nRect = nprandom.randint(1,maxCnt)
            
            nBall = 0
            
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, rotate=rotate, baseSize=baseSize, sizerange=5) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## deliberate test set for additional test 2
def generate_challenge_1_newSize2(rpath, size=[200,200], nsample=2000, rotate=True, subset = 'break_size_30_35'):
    
    componentList1 = []
    componentList2 = []
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    # class 0
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nHex = 0
        nF4 = 0
        nBall = 0
        choice = nprandom.rand()
        if choice < 0.5:
            nTri = nprandom.randint(2,6)
        else:
            nRect = nprandom.randint(2,6)
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nHex=nHex, nF4=nF4, rotate=rotate, baseSize=30, sizerange=5) )
        
    # class 1
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nHex = 0
        nF4 = 0
        nBall = 0
        
        nTri = nprandom.randint(1,4)
        nRect = nprandom.randint(1,4)
        
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nHex=nHex, nF4=nF4, rotate=rotate, baseSize=30, sizerange=5) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    
'''
=============================================================================================
counting objects
'''
def generate_count_object_data():
    generate_dataset_3_new_0(r'data\count_obj\setting1')
    generate_dataset_3_new_1(r'data\count_obj\setting2')
    generate_dataset_3_new(r'data\count_obj\setting3')
    generate_challenge_3_3_newShape(r'data\count_obj') # deliberate test 1
    generate_challenge_3_3_newSize(r'data\count_obj') # deliberate test 2

    
## 1 vs 2 objects, not included in paper
## generate data set 3, with 2 classes, simple shapes with different scales, but no rotation
## class 0: each image contains one shape
## class 1: each image contains two shapes
def generate_dataset_3(rpath, size=[200,200], nsample=2000, rotate=True, n1=1, n2=2):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            for _ in range(n1):
                choice = nprandom.randint(0,3)
                if choice == 0:
                    nTri += 1
                elif choice == 1:
                    nRect += 1
                else:
                    nBall += 1
            
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # class 1
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            for _ in range(n2):
                choice = nprandom.randint(0,3)
                if choice == 0:
                    nTri += 1
                elif choice == 1:
                    nRect += 1
                else:
                    nBall += 1
            
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## 3 vs other-than-3, setting 3
## class 0 has n1 shapes
## class 1 has more or less shapes, the number is different than n1, < n2_max
def generate_dataset_3_new(rpath, size=[200,200], nsample=2000, rotate=True, n1=3, n2_max=6):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            for _ in range(n1):
                choice = nprandom.randint(0,3)
                if choice == 0:
                    nTri += 1
                elif choice == 1:
                    nRect += 1
                else:
                    nBall += 1
            
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # class 1
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            n2 = nprandom.randint(1, n1)
            if nprandom.rand() < 0.5:
                n2 = nprandom.randint(n1+1, n2_max)
            
            for _ in range(n2):
                choice = nprandom.randint(0,3)
                if choice == 0:
                    nTri += 1
                elif choice == 1:
                    nRect += 1
                else:
                    nBall += 1
            
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## 3 vs other-than-3, setting 1
# class 0 has n1 balls
# class 1 has more or less balls, the number is different than n1, < n2_max
# only ball shape in this experiment
def generate_dataset_3_new_0(rpath, size=[200,200], nsample=2000, rotate=True, n1=3, n2_max=6):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = n1
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # class 1
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            n2 = nprandom.randint(1, n1)
            if nprandom.rand() < 0.5:
                n2 = nprandom.randint(n1+1, n2_max)
            nBall = n2
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## 3 vs other-than-3, setting 2
# class 0 has n1 shapes
# class 1 has more or less shapes, the number is different than n1, < n2_max
# only one type of shape in each image example
def generate_dataset_3_new_1(rpath, size=[200,200], nsample=2000, rotate=True, n1=3, n2_max=6):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            choice = nprandom.randint(0,3)
            if choice == 0:
                nTri = n1
            elif choice == 1:
                nRect = n1
            else:
                nBall = n1
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # class 1
        for isample in range(nsample):
            nTri = 0
            nRect = 0
            nBall = 0
            n2 = nprandom.randint(1, n1)
            if nprandom.rand() < 0.5:
                n2 = nprandom.randint(n1+1, n2_max)
            choice = nprandom.randint(0,3)
            if choice == 0:
                nTri = n2
            elif choice == 1:
                nRect = n2
            else:
                nBall = n2
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall, rotate=rotate) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## add triangle, square, ball
def addShape(imageF, componentList, result, rotate=True, choice=None, pos=None, size=None, color=None):
    im = Image.open(imageF)
    W,H = im.size
    obj = None
    
    if choice is None:
        choice = nprandom.randint(0,3)
        
    while obj is None:
        pos1 = [nprandom.randint(15,W-15), nprandom.randint(15,H-15)] if pos is None else pos
        size1 = nprandom.randint(20,30) if size is None else size
        color1 = nprandom.randint(128,255) if color is None else color
        
        if choice == 0:
            obj = drawTriangle(im, componentList, pos1, size1, color1, equilateral=True, rotate=rotate)
        elif choice == 1:
            obj = drawRectangule(im, componentList, pos1, size1, color1, rotate=rotate)
        else:
            obj = drawEllipse(im, componentList, pos1, size1, color1)
    
    componentList.append(obj)
    
    im.save(result, "PNG")
    return componentList

## deliberate test set 1
def generate_challenge_3_3_newShape(rpath=r'count_obj', size=[200,200], nsample=2000, rotate=True, n1=3, n2_max=6):
    subset = 'Deliberate_test_1'
    
    componentList1 = []
    componentList2 = []
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    
    # class 0
    for isample in range(nsample):
        nF4 = 0
        nF2 = 0
        nHex = 0
        for _ in range(n1):
            choice = nprandom.randint(0,3)
            if choice == 0:
                nF4 += 1
            elif choice == 1:
                nF2 += 1
            else:
                nHex += 1
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, 0, 0, nBall=0, nF4=nF4, nF2=nF2,  nHex=nHex, rotate=rotate) )
        
    # class 1
    for isample in range(nsample):
        nF4 = 0
        nF2 = 0
        nHex = 0
        n2 = nprandom.randint(1, n1)
        if nprandom.rand() < 0.5:
            n2 = nprandom.randint(n1+1, n2_max)
        
        for _ in range(n2):
            choice = nprandom.randint(0,3)
            if choice == 0:
                nF4 += 1
            elif choice == 1:
                nF2 += 1
            else:
                nHex += 1
        
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, 0, 0, nBall=0, nF4=nF4, nF2=nF2,nHex=nHex, rotate=rotate) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    
## deliberate test set 2
def generate_challenge_3_3_newSize(rpath=r'count_obj', size=[200,200], nsample=2000, rotate=True, n1=3, n2_max=6):
    subset = 'Deliberate_test_2'
    
    componentList1 = []
    componentList2 = []
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    
    # class 0
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nBall = 0
        for _ in range(n1):
            choice = nprandom.randint(0,3)
            if choice == 0:
                nTri += 1
            elif choice == 1:
                nRect += 1
            else:
                nBall += 1
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, rotate=rotate, baseSize=30) )
        
    # class 1
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nBall = 0
        n2 = nprandom.randint(1, n1)
        if nprandom.rand() < 0.5:
            n2 = nprandom.randint(n1+1, n2_max)
        
        for _ in range(n2):
            choice = nprandom.randint(0,3)
            if choice == 0:
                nTri += 1
            elif choice == 1:
                nRect += 1
            else:
                nBall += 1
        
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, rotate=rotate, baseSize=30) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    
    
'''
=============================================================================================
global symmetry
'''

def generate_global_symmetric_data():
    generate_dataset_4(r'data\symmetry_global\ds1',size=[200,200],nsample=4000) # A1, B1, C1
    generate_challenge_4(r'data\symmetry_global\ds1', 'train') # D1(A1)
    generate_challenge_4(r'data\symmetry_global\ds1', 'valid') # D1(B1)
    generate_dataset_4_deliberate(r'data\symmetry_global\ds2', r'data\symmetry_global\ds1') # A2, B2
    generate_challenge_4_d1(r'data\symmetry_global\ds2', 'train') # D2(A2)
    generate_challenge_4_d1(r'data\symmetry_global\ds2', 'valid') # D2(B2)
    generate_dataset_4_deliberate(r'data\symmetry_global\ds3', r'data\symmetry_global\ds2') # A3, B3
    generate_challenge_4_d2(r'data\symmetry_global\ds3') # D3(A3)
    generate_challenge_4_final(r'data\symmetry_global\ds4', 4000, subset = 'test') # A4
    generate_challenge_4_d3(r'data\symmetry_global\ds4', r'data\symmetry_global\ds3') # A3+A4, B3+B4
    generate_challenge_4_final_newShape(r'data\symmetry_global\ds4', 4000, subset = 'test_newshape') # C4
    
## add Hex, F4, F2
def addShape_new(imageF, componentList, result, rotate=True, choice=None, pos=None, size=None, color=None):
    im = Image.open(imageF)
    W,H = im.size
    obj = None
    
    if choice is None:
        choice = nprandom.randint(0,3)
        
    while obj is None:
        pos1 = [nprandom.randint(15,W-15), nprandom.randint(15,H-15)] if pos is None else pos
        size1 = nprandom.randint(20,30) if size is None else size
        color1 = nprandom.randint(128,255) if color is None else color
        
        if choice == 0:
            obj = drawHexagram(im, componentList, pos1, size1, color1, rotate=rotate)
        elif choice == 1:
            obj = drawFlower4(im, componentList, pos1, size1, color1, rotate=rotate)
        else:
            obj = drawFlower2(im, componentList, pos1, size1, color1, rotate=rotate)
    
    componentList.append(obj)
    
    im.save(result, "PNG")
    return componentList

def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n):
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result)) 
    return result
    
def testDrawSymm():
    import matplotlib.pyplot as plt
    W = 200
    H = 200
    im = Image.new('L', (W, H))

    draw = ImageDraw.Draw(im)
    
    vertices = []
    x_list = []
    y_list = []
    
    # pdb.set_trace()
    
    nPT = nprandom.randint(1, 11)
    for iPT in range(nPT):
        y = nprandom.randint(1, H)
        x = nprandom.randint(20, W/2-20)
        x_list.append(x)
        y_list.append(y)
    sID = np.argsort(y_list)
    y_list = [y_list[_] for _ in sID]
    x_list = [x_list[_] for _ in sID]
    
    vertices = [(x,y) for x,y in zip(x_list,y_list)]
    
    # draw.polygon(vertices, fill = fill)
    mid_y = np.sort(nprandom.randint(1, H, size=(nprandom.randint(0, 3),)))
    mid_x = W/2
    
    if len(mid_y) > 0:
        vertices.append((mid_x,mid_y[-1]))
    
    ## append symmetric points
    for x,y in zip(reversed(x_list),reversed(y_list)):
        vertices.append((W-x,y))
        
    if len(mid_y) > 0:
        vertices.append((mid_x,mid_y[0]))
    
    ## draw bezier curve
    if nprandom.rand() < 0.5:
        ts = [t/100.0 for t in range(101)]
        mid = len(vertices) / 2
        if len(mid_y) > 0:
            bezier = make_bezier([vertices[-1]]+vertices[:mid])
            points = bezier(ts)
            leftP = list(points)
            for pt in reversed(leftP):
                points.append((W-pt[0], pt[1]))
        else:
            bezier = make_bezier(vertices)
            points = bezier(ts)
        old_vert = vertices
        vertices = points
        
    draw.polygon(vertices, fill = 200)
    
    ## draw heart
    # ts = [t/100.0 for t in range(101)]
    # xys = [(50, 100), (80, 80), (100, 50)]
    # bezier = make_bezier(xys)
    # points = bezier(ts)

    # xys = [(100, 50), (100, 0), (50, 0), (50, 35)]
    # bezier = make_bezier(xys)
    # points.extend(bezier(ts))

    # xys = [(50, 35), (50, 0), (0, 0), (0, 50)]
    # bezier = make_bezier(xys)
    # points.extend(bezier(ts))

    # xys = [(0, 50), (20, 80), (50, 100)]
    # bezier = make_bezier(xys)
    # points.extend(bezier(ts))

    # draw.polygon(points, fill = 200)
    

    plt.imshow(im, cmap='gray')
    plt.show()

    #write to file
    im.save('test.png', "PNG")

def drawSymmetricImage(result, size, fill):
    im = Image.new('L', size)
    W, H = size

    draw = ImageDraw.Draw(im)
    
    vertices = []
    x_list = []
    y_list = []
    
    # pdb.set_trace()
    if nprandom.rand() < 0.5:
        nPT = nprandom.randint(2, 11) # 1->2
        for iPT in range(nPT):
            y = nprandom.randint(5, H-5)
            x = nprandom.randint(20, W/2-20)
            x_list.append(x)
            y_list.append(y)
        sID = np.argsort(y_list)
        y_list = [y_list[_] for _ in sID]
        x_list = [x_list[_] for _ in sID]
        
        vertices = [(x,y) for x,y in zip(x_list,y_list)]
        
        mid_y = np.sort(nprandom.randint(5, H-5, size=(nprandom.randint(0, 3),)))
        mid_x = W/2
        
        if len(mid_y) > 0:
            vertices.append((mid_x,mid_y[-1]))
        
        ## append symmetric points
        for x,y in zip(reversed(x_list),reversed(y_list)):
            vertices.append((W-x,y))
            
        if len(mid_y) > 0:
            vertices.append((mid_x,mid_y[0]))
        
        ## draw bezier curve or straight edge
        if nprandom.rand() < 0.5:
            ts = [t/100.0 for t in range(101)]
            mid = len(vertices) / 2
            if len(mid_y) > 0:
                bezier = make_bezier([vertices[-1]]+vertices[:mid])
                points = bezier(ts)
                leftP = list(points)
                for pt in reversed(leftP):
                    points.append((W-pt[0], pt[1]))
            else:
                bezier = make_bezier(vertices)
                points = bezier(ts)
            # old_vert = vertices
            vertices = points
        draw.polygon(vertices, fill = fill)
    else: # disjoint symmetry
        nPT = nprandom.randint(3, 11)
        for iPT in range(nPT):
            y = nprandom.randint(5, H-5)
            x = nprandom.randint(20, W/2-20)
            x_list.append(x)
            y_list.append(y)
        
        vertices = [(x,y) for x,y in zip(x_list,y_list)]
        
        if nprandom.rand() < 0.5:
            ts = [t/100.0 for t in range(101)]
            bezier = make_bezier(vertices)
            vertices = bezier(ts)
            
        draw.polygon(vertices, fill = fill)
        
        # flip copy left to right
        for x in range(0,W/2):
            for y in range(0,H):
                im.putpixel((W-1-x,y), im.getpixel((x,y)))

    #write to file
    im.save(result, "PNG")
    
def drawAsymmetricImage(result, size, fill):
    im = Image.new('L', size)
    W, H = size

    draw = ImageDraw.Draw(im)
    
    vertices = []
    x_list = []
    y_list = []
    
    # pdb.set_trace()
    if nprandom.rand() < 0.5:
        nPT = nprandom.randint(2, 11)
        for iPT in range(nPT):
            y = nprandom.randint(5, H-5)
            x = nprandom.randint(20, W/2-20)
            x_list.append(x)
            y_list.append(y)
        sID = np.argsort(y_list)
        y_list = [y_list[_] for _ in sID]
        x_list = [x_list[_] for _ in sID]
        
        vertices = [(x,y) for x,y in zip(x_list,y_list)]
        
        mid_y = np.sort(nprandom.randint(5, H-5, size=(nprandom.randint(0, 3),)))
        mid_x = W/2
        
        if len(mid_y) > 0:
            vertices.append((mid_x,mid_y[-1]))
        
        ## append asymmetric points
        nPT = nprandom.randint(2, 10)
        x_list = []
        y_list = []
        for iPT in range(nPT):
            y = nprandom.randint(5, H-5)
            x = nprandom.randint(W/2+20, W-20)
            x_list.append(x)
            y_list.append(y)
        sID = np.argsort(y_list)
        
        y_list = [y_list[_] for _ in sID]
        x_list = [x_list[_] for _ in sID]
        
        for x,y in zip(reversed(x_list),reversed(y_list)):
            vertices.append((x,y))
        
        if len(mid_y) > 0:
            vertices.append((mid_x,mid_y[0]))
        
        ## draw bezier curve
        if nprandom.rand() < 0.5:
            ts = [t/100.0 for t in range(101)]
            bezier = make_bezier(vertices)
            vertices = bezier(ts)
            
        draw.polygon(vertices, fill = fill)
    else: # disjoint assymetry
        nPT = nprandom.randint(3, 11)
        for iPT in range(nPT):
            y = nprandom.randint(5, H-5)
            x = nprandom.randint(20, W/2-20)
            x_list.append(x)
            y_list.append(y)
        
        vertices = [(x,y) for x,y in zip(x_list,y_list)]
        
        if nprandom.rand() < 0.5:
            ts = [t/100.0 for t in range(101)]
            bezier = make_bezier(vertices)
            vertices = bezier(ts)
            
        draw.polygon(vertices, fill = fill)
        
        # draw right
        x_list = []
        y_list = []
        nPT = nprandom.randint(3, 11)
        for iPT in range(nPT):
            y = nprandom.randint(5, H-5)
            x = nprandom.randint(W/2+20, W)
            x_list.append(x)
            y_list.append(y)
        
        vertices = [(x,y) for x,y in zip(x_list,y_list)]
        
        if nprandom.rand() < 0.5:
            ts = [t/100.0 for t in range(101)]
            bezier = make_bezier(vertices)
            vertices = bezier(ts)
            
        draw.polygon(vertices, fill = fill)

    #write to file
    im.save(result, "PNG")

## generate data set 4, with 2 classes, simple shapes with different scales
## class 0: each image contains symmetric image
## class 1: each image contains 
def generate_dataset_4(rpath, size=[200,200], nsample=2000):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        # class 0
        for isample in range(nsample):
            drawSymmetricImage(os.path.join(path0, str(isample)+'.png'), size, nprandom.randint(128, 255))
            
        # class 1
        for isample in range(nsample):
            drawAsymmetricImage(os.path.join(path1, str(isample)+'.png'), size, nprandom.randint(128, 255))
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        
def makeAsymmetric(imageF, result):
    im = Image.open(imageF)
    W,H = im.size
    draw = ImageDraw.Draw(im)
    
    x2 = W/2
    npImg = np.reshape(list(im.getdata()), (H, W))
    
    if nprandom.rand() < 0.5: # erase bar on right
        hist_x = np.sum(npImg[:, x2:], axis=0)
        tip_x_max = np.max(np.where(hist_x>1))
        tip_x_min = np.min(np.where(hist_x>1))
        
        x_min = nprandom.randint(tip_x_min, tip_x_max)
        x_max = nprandom.randint(tip_x_min, tip_x_max)
        if x_min > x_max:
            tmp = x_min
            x_min = x_max
            x_max = tmp
        if x_max-x_min < 5:
            x_min -= 3
            x_max += 3
        x0 = x2 + x_min
        x1 = x2 + x_max
        
        # x0 = x2 + tip_x_right - tip_x_right/3
        # x1 = x2 + tip_x_right
        
        y0 = 0
        y1 = H
        
    else: # erase bar on left
        hist_x = np.sum(npImg[:, :x2], axis=0)
        tip_x_max = np.max(np.where(hist_x>1))
        tip_x_min = np.min(np.where(hist_x>1))
        
        x_min = nprandom.randint(tip_x_min, tip_x_max)
        x_max = nprandom.randint(tip_x_min, tip_x_max)
        if x_min > x_max:
            tmp = x_min
            x_min = x_max
            x_max = tmp
        if x_max-x_min < 5:
            x_min -= 3
            x_max += 3
        x0 = x_min
        x1 = x_max
        
        # x0 = tip_x_left + (x2-tip_x_left)/3
        # x1 = tip_x_left
        y0 = 0
        y1 = H
        
    vertices = [(x0, y0), (x0,y1), (x1,y1), (x1,y0)]
        
    draw.polygon(vertices, fill=0)
    
    im.save(result, "PNG")
    
def makeSymmetric(imageF, result):
    im = Image.open(imageF)
    W,H = im.size
    draw = ImageDraw.Draw(im)
    
    if nprandom.rand() < 0.5: # flip left to right
        x0 = W/2
        x1 = W
        y0 = 0
        y1 = H
        
        vertices = [(x0, y0), (x0,y1), (x1,y1), (x1,y0)]
        
        draw.polygon(vertices, fill=0)
        
        for x in range(0,x0):
            for y in range(0,H):
                v = im.getpixel((x,y))
                im.putpixel((W-1-x,y),v)
    else: # flip right to left
        x0 = 0
        x1 = W/2
        y0 = 0
        y1 = H
        
        vertices = [(x0, y0), (x0,y1), (x1,y1), (x1,y0)]
        
        draw.polygon(vertices, fill=0)
        
        for x in range(x1,W):
            for y in range(0,H):
                v = im.getpixel((x,y))
                im.putpixel((W-1-x,y),v)
                
    if nprandom.rand() < 0.5: # strip black
        x2 = W/2
        npImg = np.reshape(list(im.getdata()), (H, W))
        hist_x = np.sum(npImg[:, :x2], axis=0)
        tip_x_max = np.max(np.where(hist_x>1))
        tip_x_min = np.min(np.where(hist_x>1))
        if tip_x_max-tip_x_min > 10:
            x_min = nprandom.randint(tip_x_min, tip_x_max)
            x_max = nprandom.randint(tip_x_min, tip_x_max)
            if x_min > x_max:
                tmp = x_min
                x_min = x_max
                x_max = tmp
            if x_max-x_min < 5:
                x_min -= 3
                x_max += 3
            x0 = x_min
            x1 = x_max
            
            y0 = 0
            y1 = H
            
            vertices = [(x0, y0), (x0,y1), (x1,y1), (x1,y0)]
            draw.polygon(vertices, fill=0)
            
            x0 = W-1-x_min
            x1 = W-1-x_max
            vertices = [(x0, y0), (x0,y1), (x1,y1), (x1,y0)]
            draw.polygon(vertices, fill=0)
        
    im.save(result, "PNG")
    
def generate_challenge_4(rpath, orig_set='train'):
    subset = orig_set+'_break'
    
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    newComp0 = []
    newComp1 = []
    org_path0 = os.path.join(rpath, orig_set, '0')
    org_path1 = os.path.join(rpath, orig_set, '1')
    
    nsample = len(os.listdir(org_path1))
    for isample in range(nsample):
        # class 0 from original class 1 by making it symmetric
        makeSymmetric(os.path.join(org_path1, str(isample)+'.png'), os.path.join(path0, str(isample)+'.png'))
        
    nsample = len(os.listdir(org_path0))
    for isample in range(nsample):
        # class 1 from original class 0 by making it asymmetric
        makeAsymmetric(os.path.join(org_path0, str(isample)+'.png'), os.path.join(path1, str(isample)+'.png'))
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    
def generate_dataset_4_deliberate(rpath, refpath,  ):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid']:
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)

        ## copy train cases
        org_path0 = os.path.join(refpath, subset, '0')
        org_path1 = os.path.join(refpath, subset, '1')
        
        n0 = len(os.listdir(org_path0))
        for isample in range(n0):
            shutil.copyfile(os.path.join(org_path0, str(isample)+'.png'), os.path.join(path0, str(isample)+'.png'))
        
        n1 = len(os.listdir(org_path1))
        for isample in range(n1):
            shutil.copyfile(os.path.join(org_path1, str(isample)+'.png'), os.path.join(path1, str(isample)+'.png'))
            
        ## copy challenge cases
        org_path0 = os.path.join(refpath, subset+'_break', '0')
        org_path1 = os.path.join(refpath, subset+'_break', '1')
        
        nsample = len(os.listdir(org_path0))
        for isample in range(nsample):
            shutil.copyfile(os.path.join(org_path0, str(isample)+'.png'), os.path.join(path0, str(n0+isample)+'.png'))
            
        nsample = len(os.listdir(org_path1))
        for isample in range(nsample):
            shutil.copyfile(os.path.join(org_path1, str(isample)+'.png'), os.path.join(path1, str(n1+isample)+'.png'))
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
    
def generate_challenge_4_d1(rpath, orig_set='train'):
    subset = orig_set+'_break'
    
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    newComp0 = []
    newComp1 = []
    refpath0 = os.path.join(rpath, orig_set, '0')
    refpath1 = os.path.join(rpath, orig_set, '1')
    
    nsample0 = len(os.listdir(refpath0))
    
    for isample in range(nsample0/2):
        scaleHalfImage(os.path.join(refpath0, str(isample)+'.png'), os.path.join(path0, str(isample)+'.png'), 2)
        
        scaleHalfImage(os.path.join(refpath0, str(isample)+'.png'), os.path.join(path1, str(isample)+'.png'), nprandom.randint(0,2))
    for isample in range(nsample0/2, nsample0):
        shapeHalfImage(os.path.join(refpath0, str(isample)+'.png'), os.path.join(path0, str(isample)+'.png'), 2)
        
        shapeHalfImage(os.path.join(refpath0, str(isample)+'.png'), os.path.join(path1, str(isample)+'.png'), nprandom.randint(0,2))
    
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    
def generate_challenge_4_d2(rpath, orig_set='train'):
    subset = orig_set+'_break'
    
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    newComp0 = []
    newComp1 = []
    refpath0 = os.path.join(rpath, orig_set, '0')
    refpath1 = os.path.join(rpath, orig_set, '1')
    
    nsample0 = len(os.listdir(refpath0))
    for isample in range(nsample0):
        shapeHalfImage_more(os.path.join(refpath0, str(isample)+'.png'), os.path.join(path0, str(isample)+'.png'), 2)
        
        shapeHalfImage_more(os.path.join(refpath0, str(isample)+'.png'), os.path.join(path1, str(isample)+'.png'), nprandom.randint(0,2))
    
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    
def generate_challenge_4_final(rpath, nsample, subset = 'final_test'):
    
    if not os.path.exists(rpath):
        os.makedirs(rpath)
        
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    newComp0 = []
    newComp1 = []
    
    misc.imsave('blank.png', np.zeros((200,200)))
    
    for isample in range(nsample):
        shapeHalfImage_more('blank.png', os.path.join(path0, str(isample)+'.png'), 2)
        
        shapeHalfImage_more('blank.png', os.path.join(path1, str(isample)+'.png'), nprandom.randint(0,2))
    
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    
## combine final test samples with d2 set and retrain model
def generate_challenge_4_d3(rpath, refpath):
    
    if not os.path.exists(rpath):
        os.makedirs(rpath)
        
    newtrainpath = os.path.join(rpath, 'test')
    if not os.path.exists(newtrainpath):
        generate_challenge_4_final(rpath, 4000, 'test')
    newvalpath = os.path.join(rpath, 'temp')
    generate_challenge_4_final(newvalpath, 4000, 'valid')
    
    
    for subset in ['train', 'valid']:
        subpath = os.path.join(rpath, subset)
        if os.path.exists(subpath):
            shutil.rmtree(subpath)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)

        ## copy d2
        org_path0 = os.path.join(refpath, subset, '0')
        n0 = len(os.listdir(org_path0))
        for isample in range(n0):
            shutil.copyfile(os.path.join(org_path0, str(isample)+'.png'), os.path.join(path0, str(isample)+'.png'))
        
        org_path1 = os.path.join(refpath, subset, '1')
        n1 = len(os.listdir(org_path1))
        for isample in range(n1):
            shutil.copyfile(os.path.join(org_path1, str(isample)+'.png'), os.path.join(path1, str(isample)+'.png'))
            
        
        ## copy new cases
        if subset == 'train':
            org_path0 = os.path.join(newtrainpath, '0')
            org_path1 = os.path.join(newtrainpath, '1')
        else:
            org_path0 = os.path.join(newvalpath, subset, '0')
            org_path1 = os.path.join(newvalpath, subset, '1')
        
        nsample = len(os.listdir(org_path0))
        for isample in range(nsample):
            shutil.copyfile(os.path.join(org_path0, str(isample)+'.png'), os.path.join(path0, str(n0+isample)+'.png'))
            
        nsample = len(os.listdir(org_path1))
        for isample in range(nsample):
            shutil.copyfile(os.path.join(org_path1, str(isample)+'.png'), os.path.join(path1, str(n1+isample)+'.png'))
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
    

## final deliberate test with new shapes (hex, F4, F2)
def generate_challenge_4_final_newShape(rpath, nsample, subset = 'final_test_newshape'):
    
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    newComp0 = []
    newComp1 = []
    
    misc.imsave('blank.png', np.zeros((200,200)))
    
    for isample in range(nsample):
        shapeHalfImage_more_newShape('blank.png', os.path.join(path0, str(isample)+'.png'), 2)
        
        shapeHalfImage_more_newShape('blank.png', os.path.join(path1, str(isample)+'.png'), nprandom.randint(0,2))
    
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    
    
    
def shapeHalfImage(old, new, pattern):
    im = Image.open(old)
    W,H = im.size
    
    npImg = np.reshape(list(im.getdata()), (H, W))
    
    maxv = np.max(npImg)
    
    randx = nprandom.randint(20, W/2-20)
    randy = nprandom.randint(20, H-20)
    
    if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
        addShape(old, [], 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
    else:
        addShape(old, [], 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
    
    tmp = Image.open('temp.png')
    result = np.reshape(list(tmp.getdata()), (H, W))
    
    if pattern == 2: #  both half
        result[:,W/2:] = result[:,0:W/2][:,::-1]
        
    elif pattern == 0: #  left half
        pass
        
    else: #  right half
        result = result[:,::-1]
    
    misc.imsave(new, result)
    
def hasOverlap(components, center, size):
    obj = Shape(center, size)
    for other in components:
        if obj.overlap(other):
            return True
    return False
    
    
def shapeHalfImage_more(old, new, pattern):
    im = Image.open(old)
    W,H = im.size
    
    npImg = np.reshape(list(im.getdata()), (H, W))
    
    maxv = np.max(npImg)
    
    count = nprandom.randint(2, 5)
    comp_list = []
    
    if pattern == 2: #  both half
        for i in range(count):
            randx = nprandom.randint(20, W/2-20)
            randy = nprandom.randint(20, H-20)
            while hasOverlap(comp_list, [randx, randy], 25):
                randx = nprandom.randint(20, W/2-20)
                randy = nprandom.randint(20, H-20)
                
            if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                comp_list = addShape(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
            else:
                comp_list = addShape(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
            old = 'temp.png'
    else:
        tmpdict = {'Triangle':0, 'Rectangle':1, 'Ellipse':2}
        for i in range(count):
            randx = nprandom.randint(20, W/2-20)
            randy = nprandom.randint(20, H-20)
            while hasOverlap(comp_list, [randx, randy], 25):
                randx = nprandom.randint(20, W/2-20)
                randy = nprandom.randint(20, H-20)
                
            if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                comp_list = addShape(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
            else:
                comp_list = addShape(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
            old = 'temp.png'
        for i in range(count): # for each shape, three possibles:
            comp_list_right = []
            rand_number = nprandom.rand()
            if rand_number < 0.33: # 1, add one random shape on other side
                randx = nprandom.randint(W/2+20, W)
                randy = nprandom.randint(20, H-20)
                while hasOverlap(comp_list_right, [randx, randy], 25):
                    randx = nprandom.randint(W/2+20, W)
                    randy = nprandom.randint(20, H-20)
                if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                    comp_list_right = addShape(old, comp_list_right, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
                else:
                    comp_list_right = addShape(old, comp_list_right, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
                old = 'temp.png'
            elif rand_number < 0.67: # 2, add one different shape on other side at sym position
                randx, randy = comp_list[i].center
                randx = W-randx
                choice = nprandom.randint(0,3) # 0:tri, 1, rect, 2, ellipse
                if tmpdict[comp_list[i].name] == choice:
                    choice = (choice+nprandom.randint(1,3))%3
                if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                    comp_list_right = addShape(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=25, color=0) # make a hole
                else:
                    comp_list_right = addShape(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=25, color=None) # add a shape
                old = 'temp.png'
            else: # 3, add one same shape on other side at sym position but with different scale
                randx, randy = comp_list[i].center
                randx = W-randx
                choice = tmpdict[comp_list[i].name]
                scale = float(nprandom.randint(3, 5))/10.0
                scale = 1.0+scale if nprandom.rand() < 0.5 else 1.0-scale
                size = comp_list[i].size * scale
                if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                    comp_list_right = addShape(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=size, color=0) # make a hole
                else:
                    comp_list_right = addShape(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=size, color=None) # add a shape
                old = 'temp.png'
                
    tmp = Image.open('temp.png')
    result = np.reshape(list(tmp.getdata()), (H, W))
    if pattern == 2:
        result[:,W/2:] = result[:,0:W/2][:,::-1]
    elif pattern == 0: #  left half
        pass
        
    else: #  right half
        result = result[:,::-1]
    
    misc.imsave(new, result)
    
def shapeHalfImage_more_newShape(old, new, pattern):
    im = Image.open(old)
    W,H = im.size
    
    npImg = np.reshape(list(im.getdata()), (H, W))
    
    maxv = np.max(npImg)
    
    count = nprandom.randint(2, 5)
    comp_list = []
    
    if pattern == 2: #  both half
        for i in range(count):
            randx = nprandom.randint(20, W/2-20)
            randy = nprandom.randint(20, H-20)
            while hasOverlap(comp_list, [randx, randy], 25):
                randx = nprandom.randint(20, W/2-20)
                randy = nprandom.randint(20, H-20)
                
            if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                comp_list = addShape_new(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
            else:
                comp_list = addShape_new(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
            old = 'temp.png'
    else:
        tmpdict = {'Hexagram':0, 'Flower4':1, 'Flower2':2}
        for i in range(count):
            randx = nprandom.randint(20, W/2-20)
            randy = nprandom.randint(20, H-20)
            while hasOverlap(comp_list, [randx, randy], 25):
                randx = nprandom.randint(20, W/2-20)
                randy = nprandom.randint(20, H-20)
                
            if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                comp_list = addShape_new(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
            else:
                comp_list = addShape_new(old, comp_list, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
            old = 'temp.png'
        for i in range(count): # for each shape, three possibles:
            comp_list_right = []
            rand_number = nprandom.rand()
            if rand_number < 0.33: # 1, add one random shape on other side
                randx = nprandom.randint(W/2+20, W)
                randy = nprandom.randint(20, H-20)
                while hasOverlap(comp_list_right, [randx, randy], 25):
                    randx = nprandom.randint(W/2+20, W)
                    randy = nprandom.randint(20, H-20)
                if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                    comp_list_right = addShape_new(old, comp_list_right, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=0) # make a hole
                else:
                    comp_list_right = addShape_new(old, comp_list_right, 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=25, color=None) # add a shape
                old = 'temp.png'
            elif rand_number < 0.67: # 2, add one different shape on other side at sym position
                randx, randy = comp_list[i].center
                randx = W-randx
                choice = nprandom.randint(0,3) # 0:tri, 1, rect, 2, ellipse
                if tmpdict[comp_list[i].name] == choice:
                    choice = (choice+nprandom.randint(1,3))%3
                if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                    comp_list_right = addShape_new(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=25, color=0) # make a hole
                else:
                    comp_list_right = addShape_new(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=25, color=None) # add a shape
                old = 'temp.png'
            else: # 3, add one same shape on other side at sym position but with different scale
                randx, randy = comp_list[i].center
                randx = W-randx
                choice = tmpdict[comp_list[i].name]
                scale = float(nprandom.randint(3, 5))/10.0
                scale = 1.0+scale if nprandom.rand() < 0.5 else 1.0-scale
                size = comp_list[i].size * scale
                if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
                    comp_list_right = addShape_new(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=size, color=0) # make a hole
                else:
                    comp_list_right = addShape_new(old, comp_list_right, 'temp.png', rotate=False, choice=choice, pos=[randx, randy], size=size, color=None) # add a shape
                old = 'temp.png'
                
    tmp = Image.open('temp.png')
    result = np.reshape(list(tmp.getdata()), (H, W))
    if pattern == 2:
        result[:,W/2:] = result[:,0:W/2][:,::-1]
    elif pattern == 0: #  left half
        pass
        
    else: #  right half
        result = result[:,::-1]
    
    misc.imsave(new, result)
    
def scaleHalfImage(old, new, pattern):
    im = Image.open(old)
    W,H = im.size
    
    # im2 = Image.new('L', (W,H))
    # draw = ImageDraw.Draw(im2)
    
    npImg = np.reshape(list(im.getdata()), (H, W))
    
    rate = float(nprandom.randint(3, 5))/10.0
    zoom = 1.0+rate if nprandom.rand()>0.5 else 1.0-rate
    
    if pattern == 2: # scale both half
        
        scIm = ndimage.interpolation.zoom(npImg, zoom, order=1)
        
        if zoom > 1:
            y = (scIm.shape[0]-H)/2
            x = (scIm.shape[1]-W)/2
            scaled = scIm[y:y+H, x:x+W]
        else:
            scaled = np.zeros((H,W))
            y = (H-scIm.shape[0])/2
            x = (W-scIm.shape[1])/2
            scaled[y:y+scIm.shape[0], x:x+scIm.shape[1]] = scIm
        ## small object could become nothing, or center of larged image is pure black
        if np.sum(scaled>0) < 5:
            drawSymmetricImage('temp.png', (W,H), nprandom.randint(128, 255))
            scaled = misc.imread('temp.png')
        
    elif pattern == 0: # scale left half
        
        scIm = ndimage.interpolation.zoom(npImg[:, :W/2], zoom, order=1)
        scaled = np.zeros((H,W))
        if zoom > 1:
            y = (scIm.shape[0]-H)/2
            scaled[:,:W/2] = scIm[y:y+H, -W/2:]
        else:
            y = (H-scIm.shape[0])/2
            scaled[y:y+scIm.shape[0],W/2-scIm.shape[1]:W/2] = scIm
        scaled[:,W/2:] = npImg[:, W/2:]
        
    else: # scale right half
        
        scIm = ndimage.interpolation.zoom(npImg[:, W/2:], zoom, order=1)
        scaled = np.zeros((H,W))
        if zoom > 1:
            y = (scIm.shape[0]-H)/2
            scaled[:,W/2:] = scIm[y:y+H, :W/2]
        else:
            y = (H-scIm.shape[0])/2
            scaled[y:y+scIm.shape[0],W/2:W/2+scIm.shape[1]] = scIm
        scaled[:,:W/2] = npImg[:, :W/2]
    
    misc.imsave(new, scaled)
    
    
'''
=============================================================================================
local symmetry
'''

def generate_local_symmetric_data():
    generate_dataset_5(r'data\symmetry_local',size=[200,200],nsample=4000)
    generate_challenge_5_newShape(r'data\symmetry_local') # deliberate test 1
    generate_challenge_5_newSize(r'data\symmetry_local') # deliberate test 2
    
    
def drawSymmetricPattern(im, components, center, size, fill, rotate=False, rescale=False):
    obj = Symmetry(center, size)
    for other in components:
        if obj.overlap(other):
            return None

    draw = ImageDraw.Draw(im)
    
    H, W = size, size
    
    vertices = []
    x_list = []
    y_list = []
    
    offset_x = center[0] - W/2
    offset_y = center[1] - H/2
    
    # pdb.set_trace()
    nPT = nprandom.randint(3, H/4) # 1->2
    y_list = nprandom.uniform(0, H, nPT)
    for iPT in range(nPT):
        x = nprandom.randint(1, W/2-5)
        x_list.append(x)
    sID = np.argsort(y_list)
    y_list = [int(round(y_list[_])) for _ in sID]
    x_list = [x_list[_] for _ in sID]
    
    vertices = [(x,y) for x,y in zip(x_list,y_list)]
    
    mid_y = np.sort(nprandom.randint(0, H, size=(nprandom.randint(0, 3),)))
    mid_x = W/2
    
    if len(mid_y) > 0:
        vertices.append((mid_x,mid_y[-1]))
    
    ## append symmetric points
    for x,y in zip(reversed(x_list),reversed(y_list)):
        vertices.append((W-x,y))
        
    if len(mid_y) > 0:
        vertices.append((mid_x,mid_y[0]))
    
    ## draw bezier curve or straight edge
    if nprandom.rand() < 0.5:
        ts = [t/100.0 for t in range(101)]
        mid = len(vertices) / 2
        if len(mid_y) > 0:
            bezier = make_bezier([vertices[-1]]+vertices[:mid])
            points = bezier(ts)
            leftP = list(points)
            for pt in reversed(leftP):
                points.append((W-pt[0], pt[1]))
        else:
            bezier = make_bezier(vertices)
            points = bezier(ts)
        # old_vert = vertices
        vertices = points
        
    ## -------
    
    
    vertices = [(x+offset_x, y+offset_y) for (x,y) in vertices]
    
    if rotate:
        angle = nprandom.uniform(-math.pi/4., math.pi/4.)
        R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        newV = np.dot(np.asarray(vertices)-center, R.T) + center
        vertices = newV.flatten().tolist()
    
    draw.polygon(vertices, fill = fill)
    
    
    return obj
    
def drawAsymmetricPattern(im, components, center, size, fill, rotate=False, rescale=False):
    obj = Asymmetry(center, size)
    for other in components:
        if obj.overlap(other):
            return None
            
    draw = ImageDraw.Draw(im)
    
    H, W = size, size
    
    vertices = []
    x_list = []
    y_list = []
    
    offset_x = center[0] - W/2
    offset_y = center[1] - H/2
    
    # pdb.set_trace()
    nPT = nprandom.randint(2, H/4)
    for iPT in range(nPT):
        y = nprandom.randint(0, H)
        x = nprandom.randint(0, W/2-5)
        x_list.append(x)
        y_list.append(y)
    sID = np.argsort(y_list)
    y_list = [y_list[_] for _ in sID]
    x_list = [x_list[_] for _ in sID]
    
    vertices = [(x,y) for x,y in zip(x_list,y_list)]
    
    mid_y = np.sort(nprandom.randint(0, H, size=(nprandom.randint(0, 3),)))
    mid_x = W/2
    
    if len(mid_y) > 0:
        vertices.append((mid_x,mid_y[-1]))
    
    ## append asymmetric points
    nPT = nprandom.randint(2, H/4-1)
    x_list = []
    y_list = []
    for iPT in range(nPT):
        y = nprandom.randint(0, H)
        x = nprandom.randint(W/2+5, W)
        x_list.append(x)
        y_list.append(y)
    sID = np.argsort(y_list)
    
    y_list = [y_list[_] for _ in sID]
    x_list = [x_list[_] for _ in sID]
    
    for x,y in zip(reversed(x_list),reversed(y_list)):
        vertices.append((x,y))
    
    if len(mid_y) > 0:
        vertices.append((mid_x,mid_y[0]))
    
    ## draw bezier curve
    if nprandom.rand() < 0.5:
        ts = [t/100.0 for t in range(101)]
        bezier = make_bezier(vertices)
        vertices = bezier(ts)
    
    vertices = [(x+offset_x, y+offset_y) for (x,y) in vertices]
    
    if rotate:
        angle = nprandom.uniform(-math.pi/4., math.pi/4.)
        R = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        newV = np.dot(np.asarray(vertices)-center, R.T) + center
        vertices = newV.flatten().tolist()
        
    draw.polygon(vertices, fill = fill)

    return obj
    
def scaleHalfPattern(npImg, pattern):
    H,W = npImg.shape
    # npImg = np.reshape(list(im.getdata()), (H, W))
    
    # rate = float(nprandom.randint(3, 5))/10.0
    rate = 0.5
    zoom = 1.+rate if nprandom.rand()>0.5 else 1.-rate
    
    if pattern == 0: # scale left half
        
        scIm = ndimage.interpolation.zoom(npImg[:, :W/2], zoom, order=1)
        scaled = np.zeros((H,W))
        if zoom > 1:
            y = (scIm.shape[0]-H)/2
            scaled[:,:W/2] = scIm[y:y+H, -W/2:]
        else:
            y = (H-scIm.shape[0])/2
            scaled[y:y+scIm.shape[0],W/2-scIm.shape[1]:W/2] = scIm
        scaled[:,W/2:] = npImg[:, W/2:]
        
    else: # scale right half
        
        scIm = ndimage.interpolation.zoom(npImg[:, W/2:], zoom, order=1)
        scaled = np.zeros((H,W))
        if zoom > 1:
            y = (scIm.shape[0]-H)/2
            scaled[:,W/2:] = scIm[y:y+H, :W/2]
        else:
            y = (H-scIm.shape[0])/2
            scaled[y:y+scIm.shape[0],W/2:W/2+scIm.shape[1]] = scIm
        scaled[:,:W/2] = npImg[:, :W/2]
    
    return scaled
    
def shapeHalfPattern(npImg, pattern):
    W,H = npImg.shape
    
    # npImg = np.reshape(list(im.getdata()), (H, W))
    
    maxv = np.max(npImg)
    
    randx = nprandom.randint(6, W/2-6)
    randy = nprandom.randint(6, H-6)
    
    old = 'old_pat.png'
    misc.imsave(old, npImg)
    
    if np.mean(npImg[randy-10:randy+10, randx-10:randx+10]) > (3*maxv/4):
        addShape(old, [], 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=12, color=0) # make a hole
    else:
        addShape(old, [], 'temp.png', rotate=False, choice=None, pos=[randx, randy], size=12, color=None) # add a shape
    
    result = misc.imread('temp.png')
    
    if pattern == 0: #  left half
        pass
        
    else: #  right half
        result = result[:,::-1]
    
    return result
    
def drawAsymmetricPattern2(im, components, center, size, fill, rotate=False, rescale=False):
    obj = drawSymmetricPattern(im, components, center, size, fill, rotate=rotate, rescale=rescale)
    if obj is None:
        return obj
    obj.name = 'Asymmetry'
    # pdb.set_trace()
    H,W = im.size
    npImg = np.reshape(list(im.getdata()), (H, W))
    
    npPat = npImg[center[1]-size/2:center[1]+size/2, center[0]-size/2:center[0]+size/2]
    lr = 0 if nprandom.rand() < 0.5 else 1
    if nprandom.rand() < 0.5:
        newPat = scaleHalfPattern(npPat, lr)
    else:
        newPat = shapeHalfPattern(npPat, lr)

    for x in range(center[0]-size/2, center[0]+size/2):
        for y in range(center[1]-size/2, center[1]+size/2):
            im.putpixel((x,y), newPat[y-(center[1]-size/2),x-(center[0]-size/2)])
        
    return obj
    
## mix of symmetric and asymmetric shapes in images
def generate_dataset_5(rpath, size=[200,200], nsample=4000, rotate=False):
    if os.path.exists(rpath):
        shutil.rmtree(rpath)
    os.makedirs(rpath)
    
    for subset in ['train', 'valid', 'test']:
        componentList1 = []
        componentList2 = []
        subpath = os.path.join(rpath, subset)
        os.mkdir(subpath)
        path0 = os.path.join(subpath, '0')
        path1 = os.path.join(subpath, '1')
        os.mkdir(path0)
        os.mkdir(path1)
        
        
        # class 0
        for isample in range(nsample):
            nTri = nprandom.randint(0,3)
            nRect = nprandom.randint(0,3)
            nBall = nprandom.randint(0,3)
            nSym = nprandom.randint(1,3)
            
            componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nSym=nSym, rotate=rotate, baseSize=30, sizerange=10) )
            
        # class 1
        for isample in range(nsample):
            nTri = nprandom.randint(0,2)
            nRect = nprandom.randint(0,2)
            nBall = nprandom.randint(0,2)
            nAsym = nprandom.randint(1,4)
            
            componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nSym=nSym, nAsym=nAsym, rotate=rotate, baseSize=30, sizerange=10) )
            
        # pack images to training datafile
        packImages(subpath, os.path.join(rpath, subset+'.data'))
        np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
        
## new shape of symmetric patterns
## draw new asymmetric patterns
def generate_challenge_5_newShape(rpath, size=[200,200], nsample=2000, rotate=False):
    subset = 'break_shape'
    
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    
    componentList1 = []
    componentList2 = []
    
    # class 0
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nBall = 0
        nSym = 0
        nF4 = nprandom.randint(0,4)
        nHex = nprandom.randint(0,4)
        nF2 = nprandom.randint(0,4)
        if nF4+nHex+nF2==0:
            nF4 = nprandom.randint(1,3)
            nHex = nprandom.randint(1,3)
            nF2 = nprandom.randint(1,3)
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nSym=nSym, nF4=nF4, nF2=nF2, nHex=nHex, rotate=rotate, baseSize=30, sizerange=10) )
        
    # class 1
    for isample in range(nsample):
        nTri = 0
        nRect = 0
        nBall = 0
        nSym = 0
        nF4 = nprandom.randint(0,3)
        nHex = nprandom.randint(0,3)
        nF2 = nprandom.randint(0,3)
        nAsym = nprandom.randint(1,4)
        
        ## note: set newAsym in drawOneImage function
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nAsym=nAsym, nSym=nSym, nF4=nF4, nF2=nF2, nHex=nHex, rotate=rotate, baseSize=30, sizerange=10, newAsym=True) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    

## new size of symmetric patterns
def generate_challenge_5_newSize(rpath, size=[200,200], nsample=2000, rotate=False):
    subset = 'break_size'
    
    subpath = os.path.join(rpath, subset)
    os.mkdir(subpath)
    path0 = os.path.join(subpath, '0')
    path1 = os.path.join(subpath, '1')
    os.mkdir(path0)
    os.mkdir(path1)
    
    componentList1 = []
    componentList2 = []
    
    # class 0
    for isample in range(nsample):
        nTri = nprandom.randint(0,2)
        nRect = nprandom.randint(0,2)
        nBall = nprandom.randint(0,2)
        nSym = nprandom.randint(1,3)
        nF4 = 0
        nHex = 0
        nF2 = 0
        
        componentList1.append( drawOneImage(os.path.join(path0, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nSym=nSym, nF4=nF4, nF2=nF2, nHex=nHex, rotate=rotate, baseSize=40, sizerange=5) )
        
    # class 1
    for isample in range(nsample):
        nTri = nprandom.randint(0,2)
        nRect = nprandom.randint(0,2)
        nBall = nprandom.randint(0,2)
        nAsym = nprandom.randint(1,3)
        nF4 = 0
        nHex = 0
        nF2 = 0
        
        
        componentList2.append( drawOneImage(os.path.join(path1, str(isample)+'.png'), size, nTri, nRect, nBall=nBall, nAsym=nAsym, nF4=nF4, nF2=nF2, nHex=nHex, rotate=rotate, baseSize=40, sizerange=5) )
        
    # pack images to training datafile
    packImages(subpath, os.path.join(rpath, subset+'.data'))
    np.save(os.path.join(rpath, subset+'_component.npy'), [componentList1, componentList2])
    