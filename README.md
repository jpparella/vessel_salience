# vessel_algorithm

Projeto de processamento de imagens utilizando PYTHON - Criação de métrica para avaliação de erros e problemas de segmentação em regiões com baixo contraste e algoritmo para gerar regiões de baixo contraste para treino de redes neurais.

## 🚀 Começando

Essas instruções permitirão que você obtenha uma cópia do projeto em operação na sua máquina local para fins de desenvolvimento e teste.

### 📋 Pré-requisitos


```
PYTHON 3.12
matplotlib
numpy
pilimg
time
cv2
skimage
scipy
shapely
pyvane(included)
networkx 
natsort 
oiffile 
czifile 
scikit-image 
```


## ⚙️ Executando os testes


Para executar os testes pode ser feito o passo a passo:


### 🔩 Geração de imagens (image_augmentation)

Para executar é necessário primeiro ter a imagem normal e uma imagem label, então deve ser informado o caminho para elas no seguinte local.
Observação: é necessário que a imagem de label seja binária, caso não seja pode causar problemas, mas talvez as 3 linhas sequentes possam resolver, caso o algoritmo apresente erro ou não entregue algo corretamente, verifique o tipo de imagem.
![img_1](\img_examples\img_1.png)

Após definir a imagem que será modificada, é preciso definir qual os parâmetros utilizados para modificar a imagem, sendo eles destacados em VERDE na imagem:
Observação: Caso não saiba quais parâmetros utilizar, em vermelho estão destacadas linhas de código responsável por exibir alguns dados que podem ajudar a definir os parâmetros, como por exemplo o intervalo de tamanho para região de queda deve estar dentro do tamanho máximo do vaso, sendo menor que esse tamanho. Para descobrir quantos vasos contem na imagem pode ser feito o comando "len()" na lista "tamanhoVasos". Essa lista contém todos os vasos, mas nem todos irão satisfazer os requisitos para que seja feita a modificação nele, então talvez seja necessário colocar valores menores que isso.
![img_2](\img_examples\img_2.png) 


#### Exemplos

Imagem de exemplo de modificação, no caso da imagem foi plotado também a identificação do centro da região, para que não seja plotada essa região, defina o parâmetro como ```highlight_center = false```
![img_3](\img_examples\img_3.png) 

### 🔩 Métrica Avaliação de imagens ()
Construindo exemplo

## 🛠️ Construído com

Mencione as ferramentas que você usou para criar seu projeto

* [matplotlib](https://matplotlib.org/) - Manipulação de imagens
* [pyvane](https://github.com/chcomin/pyvane) - Criação dos grafos e extração de dados
* [skimage](https://scikit-image.org/) - Criação do esqueleto
* [opencv](https://pypi.org/project/opencv-python/) - Manipulação de imagens / Extração de contornos
* [numpy](https://numpy.org/) - Manipulação de imagens 
* [PIL](https://python-pillow.org/) - Manipulação de imagens 
* [scipy](https://scipy.org/) - Convolução, dilatação e transformadas em geral nas imagens
* [scipy](https://scipy.org/) - Convolução, dilatação e transformadas em geral nas imagens


## ✒️ Autores


* **João Pedro Parella** - *Desenvolvedor* - [João Pedro Parella](https://github.com/jpparella)
* **Cesar Henrique Comin** - *Orientador e desenvolvedor* - [Cesar H Comin](https://github.com/chcomin)


