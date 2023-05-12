import PySimpleGUI as sg
import cv2 as cv
import io
import os.path
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle

global time_categorial_metrics , metrica_cat, time_cat, time_cat_metric
global time_bin_metrics,metrica_bin ,time_bin,time_bin_metric


global time_bin_metrics_r,metrica_bin_r ,time_bin_r,time_bin_metric_r
global time_categorial_metrics_r , metrica_cat_r, time_cat_r, time_cat_metric_r
global acuracia_1

### Le arquivos de tempo e modelos
file2_R = open("raso_timebin.txt",'r')
time_bin_metrics_r= file2_R.read()
metrica_bin_r = str(time_bin_metrics_r)
file2_R.close()

file1_R = open("raso_timecat.txt",'r')
time_categorial_metrics_r= file1_R.read()
metrica_cat_r = str(time_categorial_metrics_r)
file1_R.close()

file1 = open("time_categorial_metrics2.txt",'r')
time_categorial_metrics= file1.read()
metrica_cat = str(time_categorial_metrics)
file1.close()

file2_t = open("time_categorial2",'r')
time_cat_metric = file2_t.read()
time_cat = str(time_cat_metric)


file1_t = open("time_bin.txt",'r')
time_bin_metric = file1_t.read()
time_bin = str(time_bin_metric)

file2 = open("time_bin_metrics.txt",'r')
time_bin_metrics = file2.read()
metrica_bin = str(time_bin_metrics)
file2.close()

#le modelos
model_categorial = load_model("bestmodel_categorial.h5")
model_bin = load_model("bestmodel1.h5")
acuracia_1 = "41.935483870967744{%} accurate"
acuracia_2 = "21.534268647523221{%} accurate"
def modelo_raso2(path):  
    model=pickle.load(open('./img_model_Categorie.p','rb'))
    Categories=['caso0','caso1','caso2', 'caso3', 'caso4']
    caminho = str(path)
    try:
        img=imread(caminho)
            #plt.imshow(img)
            #plt.show()
        img_resize=resize(img,(150,150,3))
        l=[img_resize.flatten()]
        probability=model.predict_proba(l)

        for ind,val in enumerate(Categories):
            print(f'{val} = {probability[0][ind]*100}%')
            
        result = "The predicted image is : "+Categories[model.predict(l)[0]]
        print("The predicted image is : "+Categories[model.predict(l)[0]])
        
    except:
        print('imagem nao carregou')

    return str(result)

def modelo_raso1(path):  
    model=pickle.load(open('./img_model1.p','rb'))
    Categories=[ 'casoSA', 'casoCA']
    caminho = str(path)
    try:
        img=imread(caminho)
            #plt.imshow(img)
            #plt.show()
        img_resize=resize(img,(150,150,3))
        l=[img_resize.flatten()]
        probability=model.predict_proba(l)

        for ind,val in enumerate(Categories):
            print(f'{val} = {probability[0][ind]*100}%')
            
        result = "The predicted image is : "+Categories[model.predict(l)[0]]
        print("The predicted image is : "+Categories[model.predict(l)[0]])

        return str(result)


    except:
        print('imagem nao carregou')
        

    
  

def get_img_array(img_path): ##Função auxiliar para ler imagens
  """
  Input : Takes in image path as input 
  Output : Gives out Pre-Processed image
  """
  try:
    path = img_path
    img = image.load_img(path, target_size=(224,224,3))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis= 0 )
  except:
    exit(0)

  return img


def testa_modelo_bin(path):      # Função responsável para testar o modelo CNN-BINÁRIO
    class_type = {0:'Com artrose',  1 : 'Sem artrose'}

    try:    #predictions: path:- provide any image from google or provide image from all image folder
        img = get_img_array(path)

        res = class_type[np.argmax(model_categorial.predict(img))]
        print(f"The given X-Ray image is of type = {res}")
        print()

        # to display the image  
        plt.imshow(img[0]/255, cmap = "gray")
        plt.title(res)
        plt.show()
    except:
        print("imagem não carregou: %s" % path)




def testa_modelo_categorial(path):  #Função responsável pra testar o modelo CNN-CATEFGORIZADOR
           # you can add any image path
    class_type = {0:'Com artrose',  1 : 'Possible artrosis',2:'Definitive artrosis',3:'multiple artrosis',4:'serevere artrosis'}

    img = get_img_array(path)
    try:    #predictions: path:- provide any image from google or provide image from all image folder
        

        res = class_type[np.argmax(model_categorial.predict(img))]
        print(f"The given X-Ray image is of type = {res}")
        print()

        # to display the image  
        plt.imshow(img[0]/255, cmap = "gray")
        plt.title(res)
        plt.show()
    except:
        print("imagem não carregou: %s" %path)



def correlacao(pathImagem,pathCropped):

    img = cv.imread(pathImagem,0)
    img2 = img.copy()
    template = cv.imread(pathCropped,0)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF_NORMED','cv.TM_CCORR_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Aplica a math template com o método
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 255, 2)
        plt.imshow(img,cmap = 'gray')
        plt.title('Ponto detectado'), plt.xticks([]), plt.yticks([])
        plt.show()

global aux
sg.theme("DarkTeal2")

file_types= [
    ("JPEG (*.jpg)","*.jpg"),
    ("PNG (*.png)","*.png"),
    ("Todos os arquivos","*.*")
]

x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False   




'''

Requerimentos:
    pip install Pillow
    pip install opencv-contrib-python
    pip install matplotlib
    pip install imutils
    python -m pip install PySimpleGUI
imageCut - > def para corte da imagem (a imagem recortada é salva como cropped.jpg
                                        na pasta photos do projeto)
                                        OBS: o imageCut é um método pronto
        OBS: crie uma pasta Tmp ou temp no seu C: para que seja possível carregas as imagens
             as imagens dentro de uma pasta do projeto não estão funcionando por bug do Sistema Operacional
            parametros:
                x e y -> posiçoes do mouse
                event -> evento para clique do mouse
                flags -> flag base para setMouseCallback
                param -> argumento para setMouseCallback
'''
def imageCut(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping
    
    # Evento referente ao lique do mause
    if event == cv.EVENT_LBUTTONDOWN: 
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    
    elif event == cv.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
   
    elif event == cv.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2:
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]

            cv.imshow("corte", roi)
        
            cv.imwrite(f'C:/Tmp/cropped.jpg', roi)
'''
Primeira página -> menu
'''
def menu_window():
    sg.theme("DarkTeal2")
    menu_layout = [
        [sg.Text('Trabalho parte 1', font=('Verdana', 16), text_color='#ffffff')],
        [sg.Text('Realize o corte primeiro depois a correlação', text_color='#ffffff')],
        [sg.Text('_'*30, text_color='#ffffff')],
        [sg.Button('Cortar imagem', font=('Verdana', 12))],
        [sg.Button('Correlação', font=('Verdana', 12))],
        [sg.Button('Testar CNN', font=('Verdana', 12))],
        [sg.Button('Testar RASO', font=('Verdana', 12))],
        [sg.Button('Sair', font=('Verdana', 12), button_color='#eb4034')]
    ]
    menu_window = sg.Window('Menu', menu_layout, element_justification='c')
    while True:
        event, values = menu_window.read()
        if event == 'Sair' or event == sg.WIN_CLOSED:
            break
        elif event == 'Cortar imagem':
            menu_window.close()
            view_img_window()
        elif event == 'Correlação':
            menu_window.close()
            correlacao_window()
        elif event == 'Testar CNN':
           view_img_window_cnn_menu()
        elif event == 'Testar RASO':
            view_img_window_raso_menu()
    menu_window.close()
'''
Segunda página
'''
def view_img_window(): 
    sg.theme("DarkTeal2")
    '''
        Layout do menu
    '''
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image file"),
            sg.Input(size=(25,1), key= "-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Submit"),
            sg.Button("Cortar imagem"),
            sg.Button("Menu"),
            
        ]
    ]
    img_view_window = sg.Window("Visualizador de imagens", layout)
    '''
    Loop para rodar a aplicação da segunda página
    '''
    while True: 
        event, values = img_view_window.read()
        if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break
        elif event == "Submit": #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                aux = image.save(bio,format="PNG")
                image.save(bio,format="PNG")
                img_view_window["-IMAGE-"].update(data=bio.getvalue())

        elif event == "Cortar imagem": #Botão que libera pra cortar a imagem
                filename = values["-FILE-"]
                img = cv.imread(filename)
                #imS = cv.resize(img, (960, 540))
                #img = imS

                cv.namedWindow(filename)
                cv.setMouseCallback(filename, imageCut) #Chama imagemcut para cortar com o mouse
                
                global oriImage
                oriImage = img.copy()
                if not cropping: #Mostra a imagem sem corte
                    cv.imshow(filename, oriImage)
                if cropping:
                    cv.rectangle(oriImage, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2) #mostra a imagem cortada
        elif event == "Menu": #Volta pro menu
            img_view_window.close()
            menu_window()

    img_view_window.close()


'''
Terceira página
'''
def correlacao_window():
    sg.theme("DarkTeal2")
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Imagem Original"),
            sg.Input(size=(25,1), key= "-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Submit"),
        ],
        [sg.Image(key="-IMAGE2-")],
        [
            sg.Text("Imagem Cortada"),
            sg.Input(size=(25,1), key= "-FILE2-"),
            sg.FileBrowse(file_types=file_types,initial_folder='photos'),
            sg.Button("Submit Crop"),
        ],
        [ 
            sg.Button("Menu"),
            sg.Button("Mach")
            
        ]
    ]
    correlacao_window = sg.Window("Correlação", layout)
    '''
    Loop da aplicação da terceira página
    '''
    global caminho
    global caminho1

    while True: 
       event, values = correlacao_window.read()
       caminho = values["-FILE-"]
       caminho2 = values["-FILE2-"]
       if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break
       elif (event == "Submit"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                aux = image.save(bio,format="PNG")
                image.save(bio,format="PNG")
                correlacao_window["-IMAGE-"].update(data=bio.getvalue())
       elif (event == "Submit Crop"): #Botão submit que mostra o template da imagem
            filename2 = values["-FILE2-"]
            if os.path.exists(filename2):
                image = Image.open(values["-FILE2-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                image.save(bio,format="PNG")
                correlacao_window["-IMAGE2-"].update(data=bio.getvalue())
       elif event == ("Mach"):
                if (values["-FILE2-"]) and (values["-FILE2-"]):
                        correlacao(values["-FILE-"],values["-FILE2-"])
                else:
                    pass
       elif event == "Menu": #Volta pro menu
            correlacao_window.close()
            menu_window()



'''
Quarta pagina - menu CNN
'''
def view_img_window_cnn_menu(): 
    sg.theme("DarkTeal2")
    '''
        Layout do menu
    '''
    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(key = '-RES-')],
        [
            sg.Text("Escolha qual voce quer testar"),
            sg.Button("Testar_bin"),
            sg.Button("Testar_categorico"),
            sg.Button("Menu"),
            
        ]
    ]
    view_img_window_cnn_menu = sg.Window("Teste de rede neural", layout)
    '''
    Loop para rodar a aplicação da segunda página
    '''
    while True: 
        event, values = view_img_window_cnn_menu.read()
        if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break

        elif event == "Testar_bin": #Botão que libera pra cortar a imagem
              view_img_window_cnn_menu.close()
              cnn_window_bin()

        elif event == "Testar_categorico": #Botão que libera pra cortar a imagem
            view_img_window_cnn_menu.close()
            cnn_window_categorial()
                   
           
        elif event == "Menu": #Volta pro menu
            view_img_window_cnn_menu.close()
            menu_window()

    view_img_window_cnn_menu.close()



def cnn_window_bin():
    sg.theme("DarkTeal2")
    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(metrica_cat)],
        [sg.Text(' %s segundos' %time_bin)],
        [
            sg.Text("Imagem Original"),
            sg.Input(size=(25,1), key= "-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Submit"),
        ],
        [ 
            sg.Button("Menu"),
            sg.Button("Testar classificador")
            
        ]
    ]
    cnn_window_bin = sg.Window("Correlação", layout)
    '''
    Loop da aplicação da terceira página
    '''
    global caminho
    global caminho1

    while True: 
       event, values = cnn_window_bin.read()
       caminho = values["-FILE-"]
       if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break
       elif (event == "Submit"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                aux = image.save(bio,format="PNG")
                image.save(bio,format="PNG")
                cnn_window_bin["-IMAGE-"].update(data=bio.getvalue())

       elif (event == "Testar classificador"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            file2 = str(values["-FILE-"])
            testa_modelo_bin(file2)  
       elif event == "Menu": #Volta pro menu
            cnn_window_bin.close()
            menu_window()




def cnn_window_categorial():
    sg.theme("DarkTeal2")
    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(metrica_cat)],
        [sg.Text(' %s segundos' %time_cat)],
        [
            sg.Text("Imagem Original"),
            sg.Input(size=(25,1), key= "-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Submit"),
        ],
        [ 
            sg.Button("Menu"),
            sg.Button("Testar classificador")
            
        ]
    ]
    cnn_window_categorial = sg.Window("Correlação", layout)
    '''
    Loop da aplicação da terceira página
    '''
    global caminho
    global caminho1

    while True: 
       event, values = cnn_window_categorial.read()
       caminho = values["-FILE-"]
       if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break
       elif (event == "Submit"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                aux = image.save(bio,format="PNG")
                image.save(bio,format="PNG")
                cnn_window_categorial["-IMAGE-"].update(data=bio.getvalue())

       elif (event == "Testar classificador"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            file2 = str(values["-FILE-"])
            testa_modelo_categorial(file2)  
       elif event == "Menu": #Volta pro menu
            cnn_window_categorial.close()
            menu_window()


###############################################
def view_img_window_raso_menu(): 
    sg.theme("DarkTeal2")
    '''
        Layout do menu
    '''
    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(key = '-RES-')],
        [
            sg.Text("Escolha qual voce quer testar"),
            sg.Button("Testar_bin"),
            sg.Button("Testar_categorico"),
            sg.Button("Menu"),
            
        ]
    ]
    view_img_window_raso_menu = sg.Window("Teste de rede neural", layout)
    '''
    Loop para rodar a aplicação da segunda página
    '''
    while True: 
        event, values = view_img_window_raso_menu.read()
        if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break

        elif event == "Testar_bin": #Botão que libera pra cortar a imagem
              view_img_window_raso_menu.close()
              raso_window_bin()

        elif event == "Testar_categorico": #Botão que libera pra cortar a imagem
            view_img_window_raso_menu.close()
            raso_window_categorial()
                   
           
        elif event == "Menu": #Volta pro menu
            view_img_window_raso_menu.close()
            menu_window()

    view_img_window_raso_menu.close()



def raso_window_bin():
    sg.theme("DarkTeal2")
    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(key = "-RESULT-")],
        [sg.Text(acuracia_2)],
        [sg.Text('Tempo %s segundos' %metrica_bin_r)],
        [
            sg.Text("Imagem Original"),
            sg.Input(size=(25,1), key= "-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Submit"),
        ],
        [ 
            sg.Button("Menu"),
            sg.Button("Testar classificador")
            
        ]
    ]
    raso_window_bin = sg.Window("RASO", layout)
    '''
    Loop da aplicação da terceira página
    '''
    global caminho
    global caminho1

    while True: 
       event, values = raso_window_bin.read()
       caminho = values["-FILE-"]
       if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break
       elif (event == "Submit"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                aux = image.save(bio,format="PNG")
                image.save(bio,format="PNG")
                raso_window_bin["-IMAGE-"].update(data=bio.getvalue())

       elif (event == "Testar classificador"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            file2 = str(values["-FILE-"])
            result = modelo_raso1(file2) 
            raso_window_bin["-RESULT-"].update(result) 
       elif event == "Menu": #Volta pro menu
            raso_window_bin.close()
            menu_window()




def raso_window_categorial():
    sg.theme("DarkTeal2")
    layout = [
        [sg.Image(key="-IMAGE-")],
        [sg.Text(key = "-RESULT-")],
        [sg.Text(acuracia_1)],
        [sg.Text('Tempo %s segundos' % metrica_cat_r)],
        [
            sg.Text("Imagem Original"),
            sg.Input(size=(25,1), key= "-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Submit"),
        ],
        [ 
            sg.Button("Menu"),
            sg.Button("Testar classificador")
            
        ]
    ]
    raso_window_categorial = sg.Window("Correlação", layout)
    '''
    Loop da aplicação da terceira página
    '''
    global caminho
    global caminho1

    while True: 
       event, values = raso_window_categorial.read()
       caminho = values["-FILE-"]
       if event == sg.WIN_CLOSED or event=="Exit": #Evento para fechar a aplicação
            break
       elif (event == "Submit"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(values["-FILE-"])
                image.thumbnail((500,500))
                bio = io.BytesIO()
                aux = image.save(bio,format="PNG")
                image.save(bio,format="PNG")
                raso_window_categorial["-IMAGE-"].update(data=bio.getvalue())

       elif (event == "Testar classificador"): #Botão submit que mostra o template da imagem
            filename = values["-FILE-"]
            file2 = str(values["-FILE-"])
            result = modelo_raso2(file2) 
            raso_window_categorial["-RESULT-"].update(result) 
       elif event == "Menu": #Volta pro menu
            raso_window_categorial.close()
            menu_window()
#############################################







def main():
    menu_window()
    
  
    
if __name__ == '__main__':
    main()
