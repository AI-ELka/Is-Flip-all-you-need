#!/bin/bash

# Liste complète des machines
hosts=(
# Salle 30
allemagne
angleterre
autriche
belgique
espagne
finlande
france
groenland
hollande
hongrie
irlande
islande
lituanie
malte
monaco
pologne
portugal
roumanie
suede
# Salle 31
albatros
autruche
bengali
coucou
dindon
epervier
faisan
gelinotte
hibou
harpie
jabiru
kamiche
linotte
loriol
mouette
nandou
ombrette
perdrix
quetzal
quiscale
rouloul
sitelle
traquet
urabu
verdier
# Salle 32
aerides
barlia
calanthe
diuris
encyclia
epipactis
gennaria
habenaria
isotria
ipsea
liparis
lycaste
malaxis
neotinea
oncidium
ophrys
orchis
pleione
pogonia
serapias
telipogon
vanda
vanilla
xylobium
zeuxine
# Salle 33
ain
allier
ardennes
carmor
charente
cher
creuse
dordogne
doubs
essonne
finistere
gironde
indre
jura
landes
loire
manche
marne
mayenne
morbihan
moselle
saone
somme
vendee
vosges
# Salle 34
ablette
anchois
anguille
barbeau
barbue
baudroie
brochet
carrelet
gardon
gymnote
labre
lieu
lotte
mulet
murene
piranha
raie
requin
rouget
roussette
saumon
silure
sole
thon
truite
# Salle 35
acromion
apophyse
astragale
atlas
axis
coccyx
cote
cubitus
cuboide
femur
frontal
humerus
malleole
metacarpe
parietal
perone
phalange
radius
rotule
sacrum
sternum
tarse
temporal
tibia
xiphoide
# Salle 36
bentley
bugatti
cadillac
chrysler
corvette
ferrari
fiat
ford
jaguar
lada
maserati
mazda
nissan
niva
peugeot
pontiac
porsche
renault
rolls
rover
royce
simca
skoda
venturi
volvo
)

PYTHON_SCRIPT="~/test_script.py"

# afficher la longueur de la liste des machines
echo "Nombre de machines : ${#hosts[@]}"

for h in "${hosts[@]}"; do
    echo "Connexion à $h et test du script Python ..."
    
    ssh -o StrictHostKeyChecking=accept-new martin.beaufils@$h "
        if command -v python3 >/dev/null 2>&1; then
            echo 'Python3 trouvé sur $h'
            if [ -f $PYTHON_SCRIPT ]; then
                echo 'Lancement du script $PYTHON_SCRIPT'
                python3 $PYTHON_SCRIPT
            else
                echo 'Le script $PYTHON_SCRIPT n existe pas sur $h'
            fi
        else
            echo 'Python3 n est pas installé sur $h'
        fi
    "
done
