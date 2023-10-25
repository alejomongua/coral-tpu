# Scripts and algorithms to use Google's Coral Edge TPU (USB)

## To convert a Tensorflow model:

```
$ python3 src/convert.py /path/to/tf_model.pb /path/to/output.tflite
```
## To compile a TFLite model for the Edge TPU:

```
$ edgetpu_compiler -o /output/folder -s /path/to/tflite_model.tflite
```

## To run an inference with the Edge TPU:

```
$ python3 src/classify_image.py --model /path/to/tflite_model.tflite --labels /path/to/labels --input /path/to/image.jpg
```

## Desplegando en una Raspberry pi

Partir de una imagen de Ubuntu 20.04 para poder usar python < 3.10

Instalar los requerimientos:

    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
        tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

    sudo apt update
    sudo apt upgrade -y

    sudo apt -y install libedgetpu1-std python3-pycoral python3-pip \
                        python3-dev python3-venv git

    git clone --recurse-submodules https://github.com/alejomongua/coral-tpu.git
    
    cd coral-tpu

    python3 -m venv venv

    source venv/bin/activate

    pip install --upgrade pip

    pip install -r requirements.txt

Cuando se instala pycoral usando apt, este queda instalado en /usr/lib/python3/dist-packages/pycoral. Para poder usarlo en el entorno virtual, se debe copiar el directorio pycoral a venv/lib/python3.8/site-packages

    cp -r /usr/lib/python3/dist-packages/pycoral venv/lib/python3.8/site-packages/
    cp -r /usr/lib/python3/dist-packages/tflite_runtime venv/lib/python3.8/site-packages/

Otra opción sería agregar la ruta donde quedó instalado al PYTHONPATH

    export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages

### Limitaciones:

En la Raspberry Pi no se puede instalar el compilador edgetpu_compiler, por lo que se debe compilar en otra máquina y copiar el modelo compilado a la Raspberry Pi.


