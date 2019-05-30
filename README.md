# Chess

Za izdelavo projekta sem uporabil projek:
https://github.com/maciejczyzewski/neural-chessboard

Pred uporabo projekta je potrebno:
pip3 install opencv3
pip3 install -r requirements.txt
pip3 install python-chess

Uporaba projekta:
python3 matching.py 'full/match' 'path_to_file' 'w/b'

Rezultati se shranjujejo v mapo tempImages.

full/match:
full - vzame sliko in jo najprej zazna šahovnico, sliko obreže in nato zazna figure.
match - na že obrezani sliki zazna figure

w/b:
w - na vrsti je beli
b - na vrsti je črni
Ta argument je opcijski, če ga ne podamo program le zazna pozicijo.

Med izvajanjem program v mapo tempImages shrani vmesne slike template matchinga (za debugiranje), obrezano šahovnico in končni rezultat.

V command line se na koncu izpišejo: FEN string, tabela pozicije in čas izvajanja programa.

