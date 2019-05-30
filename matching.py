import cv2 as cv
import numpy as np
import sys
import glob
import time
import os
import chess
import chess.engine
import chess.uci
from stockfish import Stockfish
#zaporedna stevilka vmesnih slik
stVmesnihSlik = 0

#sahovnico predstavimo kot tabelo
sahovnica = [['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.'], ['.','.','.','.','.','.','.','.']]

'''
e-prazno polje
P-beli kmet
N-beli konj
B-beli lovec
R-bela trdnjava
Q-bela kraljica
K-beli kralj
p-črni kmet
n-črni konj
b-črni lovec
r-črna trdnjava
q-črna kraljica
k-črni kralj
'''

def matchFound(a, b, figura):
	#Originalne slike so vedno 1200*1200 pixlov zato lahko pozicijo figure v pikslih delimo z 150 in dobimo kvadrat v katerem je figura
	x = a / 150
	y = b / 150
	x = int(x)
	y = int(y)

	#Tresholde za template matching imamo najprej visoke in jih nato spuščamo,
	#torej če imamo v nekem kvadratu že zapisano figura, pa smo jo ponovno zaznali ZAUPAMO PREJŠNJEMU UJEMANJU
	
	temp = sahovnica[y][x]
	
	if temp == '.':
		sahovnica[y][x] = figura
	

def match(threshold, img_gray, template, figura):
	global stVmesnihSlik
	#naredimo template matching z posameznim templatom čez celotno šahovnico
	w, h = template.shape[::-1]
	res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
	loc = np.where(res >= threshold)
	 
	for pt in zip(*loc[::-1]):
		#za vsako ujemanje na sivinsko sliko narišemo kvadrat
		cv.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
		#dobimo lokacijo figure (v pixlih) in jo zapišemo v tabelo
		a = pt[0] + w/2
		b = pt[1] + h/2
		a = int(a)
		b = int(b)
		matchFound(a, b, figura)
	#za vsak template izpišemo sliko vseh ujemanj (to ni nujen del kode)	
	cv.imwrite('tempImages/vmesni-%.3d.jpg' % (stVmesnihSlik+1), img_gray)

	stVmesnihSlik = stVmesnihSlik + 1;
		

def figuraMatch(img_rgb, lokacija, threshold, figura):
	
	#template matching delamo z sevinskimi slikami
	img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

	#iz lokacije templateov za posamezno figuro preberemo vse slike
	templates = []
	datoteke = glob.glob(lokacija)

	for dat in datoteke:
		print(dat)
		img_trenutni = cv.imread(dat,0)
		templates.append(img_trenutni)
	for tmp in templates:
		#za vsak template posebej naredimo template matching
		#match('threshold', 'sivinska originalna slika', 'trenutni template', 'oznaka figure')
		match(threshold, img_gray, tmp, figura)
		img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
	
#main

start = time.time()
#preberemo način uporabe
#match - imamo že obrezano sliko in želimo zaznati figure
#full - želimo obrezati sliko in zaznati figure
#drugače izpišemo navodila za uporabo
if len(sys.argv) < 2:
	print("Uporaba:")
	print("python3 matching.py full 'path_to_file' 'w-beli na vrsti b-crni na vrsi'")
	print("Samo matching uporaba:")
	print("python3 matching.py match 'path_to_file' 'w-beli na vrsti b-crni na vrsi'")
	print("Zadnji argument ni nujen, vrne najboljšo potezo v trenutni poziciji glede na to kdo je na vrsti.")
	sys.exit()
elif sys.argv[1] == 'match':
	#direktno preberemo obrezano sliko
	img_rgb = cv.imread(sys.argv[2])
elif sys.argv[1] == 'full':
	#izvedemo ukaz, ki zažene projekt za obrezovanje slike nato preberemo sliko
	str_ukaz = ''
	str_ukaz += 'python3 main.py detect --input='
	str_ukaz += sys.argv[2]
	str_ukaz += ' --output=tempImages/obrez.jpg'
	str_ukaz
	#izvedeno ukaz za obrez slike z projektom neural chesssboard
	os.system(str_ukaz)
	img_rgb = cv.imread('tempImages/obrez.jpg')
else:
	#navodila za uporabo
	print("Uporaba:")
	print("python3 matching.py full 'path_to_file' 'w-beli na vrsti b-crni na vrsi'")
	print("Samo matching uporaba:")
	print("python3 matching.py match 'path_to_file' 'w-beli na vrsti b-crni na vrsi'")
	sys.exit()


#za vsako figuro posebej naredimo template matching
'''
e-prazno polje
P-beli kmet
N-beli konj
B-beli lovec
R-bela trdnjava
Q-bela kraljica
K-beli kralj
p-črni kmet
n-črni konj
b-črni lovec
r-črna trdnjava
q-črna kraljica
k-črni kralj
'''
#figuraMatch('originalna slika', 'pot do mape z templati', 'treshold', 'oznaka za figuro')
figuraMatch(img_rgb, 'templates/BKONJ/T*.jpg', 0.85, 'N')
figuraMatch(img_rgb, 'templates/CKONJ/T*.jpg', 0.85, 'n')

figuraMatch(img_rgb, 'templates/BTRDNJAVA/T*.jpg', 0.8, 'R')
figuraMatch(img_rgb, 'templates/CTRDNJAVA/T*.jpg', 0.8, 'r')

figuraMatch(img_rgb, 'templates/CKMET/T*.jpg', 0.83, 'p')
figuraMatch(img_rgb, 'templates/BKMET/T*.jpg', 0.81, 'P')

figuraMatch(img_rgb, 'templates/CKRALJ/T*.jpg', 0.8, 'k')
figuraMatch(img_rgb, 'templates/BKRALJ/T*.jpg', 0.83, 'K')
figuraMatch(img_rgb, 'templates/BDAMA/T*.jpg', 0.78, 'Q')

figuraMatch(img_rgb, 'templates/BLOVEC/T*.jpg', 0.75, 'B')
figuraMatch(img_rgb, 'templates/CLOVEC/T*.jpg', 0.75, 'b')

figuraMatch(img_rgb, 'templates/CDAMA/T*.jpg', 0.78, 'q')

img_sah = cv.imread('templates/SAHOVNICA/sahovnica.jpg')

img_BKNB = cv.imread('templates/SAHOVNICA/BKNB.jpg')
img_BKNC = cv.imread('templates/SAHOVNICA/BKNC.jpg')
img_BJNB = cv.imread('templates/SAHOVNICA/BJNB.jpg')
img_BJNC = cv.imread('templates/SAHOVNICA/BJNC.jpg')
img_BLNB = cv.imread('templates/SAHOVNICA/BLNB.jpg')
img_BLNC = cv.imread('templates/SAHOVNICA/BLNC.jpg')
img_BTNB = cv.imread('templates/SAHOVNICA/BTNB.jpg')
img_BTNC = cv.imread('templates/SAHOVNICA/BTNC.jpg')
img_BDNB = cv.imread('templates/SAHOVNICA/BDNB.jpg')
img_BDNC = cv.imread('templates/SAHOVNICA/BDNC.jpg')
img_BENB = cv.imread('templates/SAHOVNICA/BENB.jpg')
img_BENC = cv.imread('templates/SAHOVNICA/BENC.jpg')


img_CKNB = cv.imread('templates/SAHOVNICA/CKNB.jpg')
img_CKNC = cv.imread('templates/SAHOVNICA/CKNC.jpg')
img_CJNB = cv.imread('templates/SAHOVNICA/CJNB.jpg')
img_CJNC = cv.imread('templates/SAHOVNICA/CJNC.jpg')
img_CLNB = cv.imread('templates/SAHOVNICA/CLNB.jpg')
img_CLNC = cv.imread('templates/SAHOVNICA/CLNC.jpg')
img_CTNB = cv.imread('templates/SAHOVNICA/CTNB.jpg')
img_CTNC = cv.imread('templates/SAHOVNICA/CTNC.jpg')
img_CDNB = cv.imread('templates/SAHOVNICA/CDNB.jpg')
img_CDNC = cv.imread('templates/SAHOVNICA/CDNC.jpg')
img_CENB = cv.imread('templates/SAHOVNICA/CENB.jpg')
img_CENC = cv.imread('templates/SAHOVNICA/CENC.jpg')


#naredimo sliko sahovnice

for y in range(8):
	for x in range(8):
		x_offset = x * 150
		y_offset = y * 150
		poz = x + y
		poz = poz % 2
			
		if sahovnica[x][y] == 'P':
			if poz == 0:
				img_sah[x_offset:x_offset+img_BKNB.shape[1], y_offset:y_offset+img_BKNB.shape[0]] = img_BKNB
			else:
				img_sah[x_offset:x_offset+img_BKNC.shape[1], y_offset:y_offset+img_BKNC.shape[0]] = img_BKNC
		elif sahovnica[x][y] == 'N':
			if poz == 0:
				img_sah[x_offset:x_offset+img_BJNB.shape[1], y_offset:y_offset+img_BJNB.shape[0]] = img_BJNB
			else:
				img_sah[x_offset:x_offset+img_BJNC.shape[1], y_offset:y_offset+img_BJNC.shape[0]] = img_BJNC
		elif sahovnica[x][y] == 'B':
			if poz == 0:
				img_sah[x_offset:x_offset+img_BLNB.shape[1], y_offset:y_offset+img_BLNB.shape[0]] = img_BLNB
			else:
				img_sah[x_offset:x_offset+img_BLNC.shape[1], y_offset:y_offset+img_BLNC.shape[0]] = img_BLNC
		elif sahovnica[x][y] == 'R':
			if poz == 0:
				img_sah[x_offset:x_offset+img_BTNB.shape[1], y_offset:y_offset+img_BTNB.shape[0]] = img_BTNB			
			else:
				img_sah[x_offset:x_offset+img_BTNC.shape[1], y_offset:y_offset+img_BTNC.shape[0]] = img_BTNC
		elif sahovnica[x][y] == 'Q':
			if poz == 0:
				img_sah[x_offset:x_offset+img_BDNB.shape[1], y_offset:y_offset+img_BDNB.shape[0]] = img_BDNB			
			else:
				img_sah[x_offset:x_offset+img_BDNC.shape[1], y_offset:y_offset+img_BDNC.shape[0]] = img_BDNC
		elif sahovnica[x][y] == 'K':
			if poz == 0:
				img_sah[x_offset:x_offset+img_BENB.shape[1], y_offset:y_offset+img_BENB.shape[0]] = img_BENB			
			else:
				img_sah[x_offset:x_offset+img_BENC.shape[1], y_offset:y_offset+img_BENC.shape[0]] = img_BENC		
		if sahovnica[x][y] == 'p':
			if poz == 0:
				img_sah[x_offset:x_offset+img_CKNB.shape[1], y_offset:y_offset+img_CKNB.shape[0]] = img_CKNB	
			else:
				img_sah[x_offset:x_offset+img_CKNC.shape[1], y_offset:y_offset+img_CKNC.shape[0]] = img_CKNC
				
		elif sahovnica[x][y] == 'n':
			if poz == 0:
				img_sah[x_offset:x_offset+img_CJNB.shape[1], y_offset:y_offset+img_CJNB.shape[0]] = img_CJNB
			else:
				img_sah[x_offset:x_offset+img_CJNC.shape[1], y_offset:y_offset+img_CJNC.shape[0]] = img_CJNC
		elif sahovnica[x][y] == 'b':
			if poz == 0:
				img_sah[x_offset:x_offset+img_CLNB.shape[1], y_offset:y_offset+img_CLNB.shape[0]] = img_CLNB
			else:
				img_sah[x_offset:x_offset+img_CLNC.shape[1], y_offset:y_offset+img_CLNC.shape[0]] = img_CLNC
		elif sahovnica[x][y] == 'r':
			if poz == 0:
				img_sah[x_offset:x_offset+img_CTNB.shape[1], y_offset:y_offset+img_CTNB.shape[0]] = img_CTNB
			else:
				img_sah[x_offset:x_offset+img_CTNC.shape[1], y_offset:y_offset+img_CTNC.shape[0]] = img_CTNC
		elif sahovnica[x][y] == 'q':
			if poz == 0:
				img_sah[x_offset:x_offset+img_CDNB.shape[1], y_offset:y_offset+img_CDNB.shape[0]] = img_CDNB
			else:
				img_sah[x_offset:x_offset+img_CDNC.shape[1], y_offset:y_offset+img_CDNC.shape[0]] = img_CDNC
		elif sahovnica[x][y] == 'k':
			if poz == 0:
				img_sah[x_offset:x_offset+img_CENB.shape[1], y_offset:y_offset+img_CENB.shape[0]] = img_CENB
			else:
				img_sah[x_offset:x_offset+img_CENC.shape[1], y_offset:y_offset+img_CENC.shape[0]] = img_CENC

cv.imwrite('tempImages/rezultat.jpg', img_sah)

#naredimo FEN string
boolPrazni = 0
countPrazni = 0
FEN = ''

for a in range(8):
	for b in range(8):
		if boolPrazni == 0:
			if sahovnica[a][b] == '.':
				boolPrazni = 1
				countPrazni += 1
			else:
				FEN += sahovnica[a][b]
		else:
			if sahovnica[a][b] == '.':
				countPrazni += 1
			else:
				boolPrazni = 0
				FEN += str(countPrazni)
				countPrazni = 0
				FEN += sahovnica[a][b]
	if countPrazni != 0:
		FEN += str(countPrazni)
	countPrazni = 0
	boolPrazni = 0
	if a != 7:		
		FEN += '/'
					
FEN += ' '
if len(sys.argv) == 4:
	FEN += sys.argv[3]
else:
	FEN += 'w'
FEN += ' KQkq - 0 1'

#izpisemo FEN in šahovnico
print(FEN)
board = chess.Board(FEN)
print(board)

#izpišemo pozicijo po najboljši pozeti
if len(sys.argv) == 4:
	engine = chess.engine.SimpleEngine.popen_uci("stockfish")
	limit = chess.engine.Limit(time=2.0)
	engine.play(board, limit)
	print("Pozicija po najboljši potezi:")
	print(board)

#izpišemo čas programa

end = time.time()
cas = end - start
cas = round(cas, 2)
print(cas, end='')
print(" s")


