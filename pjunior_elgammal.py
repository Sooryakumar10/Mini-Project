import imageio
import cv2
from collections import namedtuple
from random import *
from numpy import *
import math
import time


Point = namedtuple("Point", "x y")
a=randint(2,10)
img_raw=cv2.imread("G:\\14.jpg")
cv2.imshow('OP', img_raw)
cv2.waitKey(0)
print("\nOriginal Array\n", img_raw)

start=time.time()


eO = 'ORIGIN'
ep = 251
ea = 5
eb = 5
ena = randint(1, ep - 1)
enb = 1
k = 59
eg = '90,7'
epb = ekpb = ekg = nbkg=''

def valid(P):
    if P == eO:
        return True
    else:
        return (
                (P.y ** 2 - (P.x ** 3 + ea * P.x + eb)) % ep == 0 and
                0 <= P.x < ep and 0 <= P.y < ep)


def inv_mod_p(x):
    if x % ep == 0:
        raise ZeroDivisionError("Impossible inverse")
    return pow(x, ep - 2, ep)


def ec_inv(P):
    if P == eO:
        return P
    return Point(P.x, (-P.y) % ep)


def ec_add(P, Q):
    if not (valid(P) and valid(Q)):
        raise ValueError("Invalid inputs")

    if P == eO:
        result = Q
    elif Q == eO:
        result = P
    elif Q == ec_inv(P):
        result = eO
    else:
        if P == Q:
            dydx = (3 * P.x ** 2 + ea) * inv_mod_p(2 * P.y)
        else:
            dydx = (Q.y - P.y) * inv_mod_p(Q.x - P.x)
        x = (dydx ** 2 - P.x - Q.x) % ep
        y = (dydx * (P.x - x) - P.y) % ep
        result = Point(x, y)

    assert valid(result)
    return result


def calc(nat, P, Q):
    while nat != 1:
        r = ec_add(P, Q)
        if r == eO:
            P = r
        else:
            xp = r.x
            yp = r.y
            P = Point(xp, yp)
        nat = nat - 1
    if r == eO:
        temp = r
    else:
        temp = str(xp) + ',' + str(yp)
    return temp

nat = k
xp = xq = int(eg.split(',')[0])
yp = yq = int(eg.split(',')[1])
P = Point(xp, yp)
Q = Point(xq, yq)
if k == 1:
    ekg = eg
else:
    ekg = calc(nat, P, Q)

if ekg == eO:
    seed_value = randint(1, 256)
else:
    seed_value = int(ekg.split(',')[1])

pixel_array=[]

for i in img_raw:
    for j in i:
        pixel_array.append(j)
        
class mersenne_rng(object):
    def __init__(self, seed_value):
        self.state = [0] * 624
        self.f = 1812433253
        self.m = 397
        self.u = 11
        self.s = 7
        self.b = 0x9D2C5680
        self.t = 15
        self.c = 0xEFC60000
        self.l = 18
        self.index = 624
        self.lower_mask = (1 << 31) - 1
        self.upper_mask = 1 << 31

        self.state[0] = seed_value
        for i in range(1, 624):
            self.state[i] = self.int_32(self.f * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + i)

    def twist(self):
        for i in range(624):
            temp = self.int_32((self.state[i] & self.upper_mask) + (self.state[(i + 1) % 624] & self.lower_mask))
            temp_shift = temp >> 1
            if temp % 2 != 0:
                temp_shift = temp_shift ^ 0x9908b0df
            self.state[i] = self.state[(i + self.m) % 624] ^ temp_shift
        self.index = 0

    def get_random_number(self):
        if self.index >= 624:
            self.twist()
        y = self.state[self.index]
        y = y ^ (y >> self.u)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        self.index += 1
        return self.int_32(y)

    def int_32(self, number):
        return int(0xFFFFFFFF & number)
    
if __name__ == "__main__":
    rng = mersenne_rng(seed_value)
    random = []
    for i in range(3):
        random_number = rng.get_random_number()
        random.append(random_number)

    print("\nThe Random Array before updating:", random)

    for i in random:
        if (len(str(i)) == 10):
            random
        elif (len(str(i)) < 10):
            random.remove(i)
            i = str(i) + "0"
            random.append(int(i))

    print("\nThe updated random array is:", random)

xor_array = bitwise_xor(pixel_array, random)
xor_array1 = xor_array%256
print("\nThe XOR array after bitwise with random and performing mod 256 is\n", xor_array1)

print("ECC and Elgammal Encryption is on progress...")

points = []
lhs = rhs = 0
for i in range(ep):
    for j in range(ep):
        lhs = (j * j) % ep
        rhs = ((i * i * i) + (ea * i) + eb) % ep
        if lhs == rhs:
            points.append(str(i) + ',' + str(j))
points.append(eO)



nat = enb
xp = xq = int(eg.split(',')[0])
yp = yq = int(eg.split(',')[1])
P = Point(xp, yp)
Q = Point(xq, yq)
if enb == 1:
    epb = eg
else:
    epb = calc(nat, P, Q)

nat = k
if k == 1:
    ekpb = epb
else:
    if epb == eO:
        P = Q = eO
        ekpb = calc(nat, P, Q)
    else:
        xp = xq = int(epb.split(',')[0])
        yp = yq = int(epb.split(',')[1])
        P = Point(xp, yp)
        Q = Point(xq, yq)
        ekpb = calc(nat, P, Q)

nat = k*enb
if epb==eO:
    P=Q=eO
else:
    xp = xq = int(eg.split(',')[0])
    yp = yq = ep-int(eg.split(',')[1])
    P = Point(xp, yp)
    Q = Point(xq, yq)
if nat == 1:
    nbkg = eg
else:
    nbkg=calc(nat,P,Q)


op = []
for i in xor_array1:
    t1 = []
    for j in range(3):
        pm = points[i[j]]
        if pm == eO:
            P = eO
        else:
            xp = int(pm.split(',')[0])
            yp = int(pm.split(',')[1])
            P = Point(xp, yp)
        if ekpb == eO:
            Q = eO
        else:
            xq = int(ekpb.split(',')[0])
            yq = int(ekpb.split(',')[1])
            Q = Point(xq, yq)
        R = ec_add(P, Q)
        if R == eO:
            t1.append(points.index(eO))
        else:
            xr = R.x
            yr = R.y
            t = str(xr) + ',' + str(yr)
            t1.append(points.index(t))
    t2 = t1[:]
    op.append(t2)

op1 = array(op)
print("\nThe New Pixel Array after Applying ECC is\n", op1)

def nest_list(list1,rows, columns):
    result=[]
    start = 0
    end = columns
    for i in range(rows):
        result.append(list1[start:end])
        start +=columns
        end += columns
    return result

def gcd(a,b):
    if a<b:
        return gcd(b,a)
    elif a%b==0:
        return b
    else:
        return gcd(b,a%b)

def gen_key(q):
    key= randint(math.pow(2,3),q)
    while gcd(q,key)!=1:
        key=randint(math.pow(10,1),q)
    return key

def power(a,b,c):
    x=1
    y=a
    while b>0:
        if b%2==0:
            x=(x*y)%c;
        y=(y*y)%c
        b=int(b/2)
    return x%c

def encryption(msg,q,h,g):
    ct=[]
    k=gen_key(q)
    s=power(h,k,q)
    p=power(g,k,q)
    ct.append(int(msg))
    for i in range(0,1):
        ct[i]=s*ord(chr(ct[i]))
    return ct,p

def decryption(ct,p,key,q):
    pt=[]
    h=power(p,key,q)
    for i in range(0,len(ct)):
        pt.append(ord(chr(int(ct[i]/h))))
    return pt

q=randint(math.pow(10,1),math.pow(10,2))
g=randint(2,q)
key=gen_key(q)
h=power(g,key,q)

P_V=[]
fa=[]
for i in op:
    for j in i:
        x,y=encryption(j,q,h,g)
        P_V.append(y)
        fa.append(x[0])
L=len(fa)
fa1=nest_list(fa,int(L/3),3)
fa1=nest_list(fa1,img_raw.shape[0],img_raw.shape[1])
fa_=array(fa1,dtype=uint8)

cv2.imwrite('G:\\enc.jpg',fa_)
cv2.imshow('ENCOP',fa_)
cv2.waitKey(5000)
cv2.destroyWindow('ENCOP')
print('\nThe Final Encrypted Array is\n',fa_)


print('\n..............THE DECRYPTION PROCESS BEGINS..........\n')
opa=[]
y=[]
for i in range(L):
    y.append(fa[i])
    x=decryption(y,P_V[i],key,q)
    opa.append(x[0])
    y.clear()
opa=nest_list(opa,int(L/3),3)

fa2 = array(opa,dtype=uint8)
print('\nThe array after ELGAMMAL decryption is\n', fa2)

op=[]
for i in opa:
    t1=[]
    for k in range(3):
        pmkpb=points[i[k]]
        if pmkpb==eO:
            P=eO
        else:
            xp = int(pmkpb.split(',')[0])
            yp = int(pmkpb.split(',')[1])
            P = Point(xp, yp)
        if nbkg == eO:
            Q = eO
        else:
            Q = Point(int(nbkg.split(',')[0]), int(nbkg.split(',')[1]))
        R=ec_add(P,Q)
        if R == eO:
            t1.append(points.index(eO))
        else:
            xr = R.x
            yr = R.y
            t = str(xr) + ',' + str(yr)
            t1.append(points.index(t))
    t2 = t1[:]
    op.append(t2)

fa3=array(op,dtype=uint8)
print('\nThe array after ECC decryption is\n', fa3)

fa= bitwise_xor(op, random)
fa1=fa%256
print("\nThe Final decrypted array is\n",fa1)

fa1=nest_list(fa1,img_raw.shape[0],img_raw.shape[1])
fa=array(fa1,dtype=uint8)
cv2.imwrite('G:\\dec.jpg',fa)
end=time.time()
print("Duration =",((end-start)-5))
cv2.imshow('DECOP',fa)

cv2.waitKey(0)
