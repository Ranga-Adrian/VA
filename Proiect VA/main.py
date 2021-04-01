import cv2

thres = 0.65  # Pragul de detectie al obiectelor

#img = cv2.imread("lena.PNG")
# Citim captura din Video-ul
cap = cv2.VideoCapture(0)

# Setam parametrii privind dimensiunea capturii din video
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Importam clasele din coco.names
classFile = 'coco.names'

# Pregatim un vector gol pentru a pune datele din coco.names in ea
classNames = []

# Deschidem fisierul pentru citire cu rt
with open(classFile, 'rt') as f:
    # adaugam datele in vectorul classNames
    classNames = f.read().rstrip('\n').split('\n')

# MobileNet SSD este o arhitectura destinata efectuarii detectarii obiectelor.
# Folosim mobilenet ssd pentru configuratii deoarece este una dintre cele mai bune metode actuale
# ce detine o balanta buna intre viteza si acuratete, utilizand procesorul dispozitivului de pe care
# se face detectia aproape in timp real si in conditii optime
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# In acest moment avand configuratiile ne putem crea modelul

# Mai jos setam parametrii pentru preprocesarea imaginii de intrare
# DetectionModel creeaza o retea folosind fisierul cu configuratii si graph-uri antrenate,
# seteaza intrarea de preprocesare ruleaza treceri inainte si returneaza detectiile rezultate
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # Seteaza dimensiunea de intrare a imaginii
# Seteaza valoarea factorului de scala pentru fiecare cadru
net.setInputScale(1.0 / 127.5)
# Seteaza valoarea medie pentru fiecare cadru
net.setInputMean((127.5, 127.5, 127.5))
# Seteaza flag-ul pentru schimbarea primului si al ultimului canal cu 1
net.setInputSwapRB(True)

# In pasul urmator trimitem imaginea cadrului modelului nostru obtinand astfel predictiile modelului

while True:
    success, img = cap.read()  # Citeste si adauga cadrele pe rand in variabila img
    # Obtinem o caseta de incadrare a imaginii cu informatiile aferente acesteia
    # (ID-ul caseta de incadrare, confidenta) cu un prag de 50%
    classIds, confs, bbox = net.detect(img, thres)
    print(classIds, bbox)

    # In pasul urmator efectuam o cautare prin toate informatiile/variabilele obtinute in pasul anterior
    # (ID,configurari, caseta de incadrare a imaginii) folosind o singura functie de cautare (for )
    # si o functie (zip)

    if len(classIds) != 0:  # Cat timp exista un id de citit efectueaza urmatorii pasi
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

            # Cream un dreptunghi asupra imaginii img de marimea bounding box-ului de culoare verde
            # si grosime a liniei de 3
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)

            # Adaugam text pe imaginea noastra adaugand numele imaginii(este classId-1 deoarece in coco.names
            # incepe de la 1 iar vectorul classNames incepe de la 0) la coordonatele X si Y oferite de
            # bounding box la un factor de scala de 1 avand culoarea verde cu o grosime de 2
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Adaugam ca si text confidenta cu care este detectata o imaginea a unui obiect ca fiind o persoana,
            # un dulap, etc...
            cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output', img)  # afiseaza fiecare cadru ca si Output
    cv2.waitKey(1)  # asteapta o tasta
